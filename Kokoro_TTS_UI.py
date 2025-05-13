import sounddevice as sd
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from kokoro import KModel, KPipeline
import soundfile as sf
import threading
from queue import Queue

class TTSApp:
    """语音合成应用程序（线程安全版）"""
    
    def __init__(self, master):
        self.master = master
        master.title("Kokoro 语音合成器 v2.0")
        master.geometry("800x600")

        # 初始化TTS引擎
        self.tts_engine = KokoroTTS()
        self.current_audio = None
        self.task_queue = Queue()
        
        # 创建界面组件
        self._create_widgets()
        
        # 启动任务检查器
        self.master.after(100, self.process_queue)

    def _create_widgets(self):
        """创建界面组件"""
        # 音色选择
        ttk.Label(self.master, text="选择音色:").pack(pady=5)
        self.voice_var = tk.StringVar()
        
        # 生成所有可用音色选项
        self.voice_options = [
            f"zf_{i:03d}" for i in range(1, 11)
        ] + [
            f"zm_{i:03d}" for i in range(1, 11)
        ]
        
        self.voice_combobox = ttk.Combobox(
            self.master,
            textvariable=self.voice_var,
            values=self.voice_options,
            state='readonly'
        )
        self.voice_combobox.current(0)  # 设置默认选项
        self.voice_combobox.pack(pady=5)

        # 文本输入
        ttk.Label(self.master, text="输入文本:").pack(pady=5)
        self.text_input = tk.Text(self.master, height=10, wrap=tk.WORD)
        self.text_input.pack(pady=5, fill=tk.BOTH, expand=True)

        # 控制按钮
        btn_frame = ttk.Frame(self.master)
        btn_frame.pack(pady=10)
        
        self.generate_btn = ttk.Button(
            btn_frame,
            text="生成语音",
            command=self.start_generate_thread
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(
            btn_frame,
            text="播放",
            command=self.play_audio,
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(
            btn_frame,
            text="保存音频",
            command=self.save_audio,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # 进度指示
        self.progress = ttk.Progressbar(
            self.master,
            mode='indeterminate',
            length=200
        )
        
        # 状态栏
        self.status_var = tk.StringVar()
        ttk.Label(
            self.master,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        ).pack(side=tk.BOTTOM, fill=tk.X)

    def process_queue(self):
        """处理线程任务队列"""
        try:
            msg = self.task_queue.get_nowait()
            if msg == 'start':
                self.progress.pack(pady=10)
                self.progress.start()
                self.toggle_buttons(False)
            elif msg == 'done':
                self.progress.stop()
                self.progress.pack_forget()
                self.toggle_buttons(True)
                self.status_var.set("音频生成完成！")
            elif msg.startswith('error'):
                self.progress.stop()
                self.progress.pack_forget()
                self.toggle_buttons(True)
                messagebox.showerror("生成错误", msg[6:])
        except:
            pass
        finally:
            self.master.after(100, self.process_queue)

    def toggle_buttons(self, state: bool):
        """切换按钮状态"""
        state = tk.NORMAL if state else tk.DISABLED
        self.generate_btn.config(state=state)
        self.play_btn.config(state=state)
        self.save_btn.config(state=state)

    def start_generate_thread(self):
        """启动生成线程"""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("错误", "请输入文本内容！")
            return

        voice = self.voice_var.get()
        if voice not in self.voice_options:
            messagebox.showerror("错误", "无效的音色选择！")
            return

        # 创建并启动工作线程
        thread = threading.Thread(
            target=self.generate_audio,
            args=(text, voice),
            daemon=True
        )
        self.task_queue.put('start')
        thread.start()

    def generate_audio(self, text: str, voice: str):
        """后台生成音频"""
        try:
            self.current_audio = self.tts_engine.generate_speech(text, voice=voice)
            self.task_queue.put('done')
        except Exception as e:
            self.task_queue.put(f'error {str(e)}')

    def play_audio(self):
        """播放音频"""
        if self.current_audio is None:
            return

        def _play():
            try:
                self.tts_engine.play_audio(self.current_audio)
                self.status_var.set("播放完成")
            except Exception as e:
                messagebox.showerror("播放错误", str(e))

        # 在独立线程中播放
        threading.Thread(target=_play, daemon=True).start()

    def save_audio(self):
        """保存音频"""
        if self.current_audio is None:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                audio_np = self.current_audio.cpu().numpy()
                sf.write(file_path, audio_np, self.tts_engine.sample_rate)
                self.status_var.set(f"文件已保存至：{file_path}")
            except Exception as e:
                messagebox.showerror("保存错误", str(e))

class KokoroTTS:
    """语音合成核心类（优化版）"""
    
    def __init__(self, repo_id: str = 'hexgrad/Kokoro-82M-v1.1-zh'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.repo_id = repo_id
        self.sample_rate = 24000
        self._init_models()

    def _init_models(self) -> None:
        """初始化语音模型"""
        with torch.no_grad():
            self.en_pipeline = KPipeline(lang_code='a', repo_id=self.repo_id, model=False)
            self.zh_model = KModel(repo_id=self.repo_id).to(self.device).eval()
            self.zh_pipeline = KPipeline(
                lang_code='z',
                repo_id=self.repo_id,
                model=self.zh_model,
                en_callable=self._en_callable
            )

    def _en_callable(self, text: str) -> str:
        """处理特殊英文词汇"""
        custom_dict = {'Kokoro': 'kˈOkəɹO', 'Sol': 'sˈOl'}
        return custom_dict.get(text, next(self.en_pipeline(text)).phonemes)

    @staticmethod
    def _dynamic_speed(len_ps: int) -> float:
        """动态语速算法"""
        if len_ps <= 83:
            return 1.1
        if len_ps < 183:
            return (1 - (len_ps - 83) / 500) * 1.1
        return 0.88

    def generate_speech(self, text: str, voice: str) -> torch.Tensor:
        """生成语音波形"""
        with torch.no_grad():
            generator = self.zh_pipeline(text, voice=voice, speed=self._dynamic_speed)
            return next(generator).audio.clone()

    @staticmethod
    def play_audio(wav: torch.Tensor, sample_rate: int = 24000) -> None:
        """播放音频"""
        sd.play(wav.cpu().numpy(), samplerate=sample_rate)
        sd.wait()

if __name__ == "__main__":
    root = tk.Tk()
    app = TTSApp(root)
    root.mainloop()
