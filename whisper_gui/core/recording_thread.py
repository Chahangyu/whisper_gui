"""
RecordingThread class - Handles audio recording in a background thread.
"""

import os
import tempfile
import pyaudio
import wave
from PyQt6.QtCore import QThread, pyqtSignal

class RecordingThread(QThread):
    """오디오 녹음을 위한 스레드 클래스"""
    finished = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    
    def __init__(self, max_duration=10):
        super().__init__()
        self.max_duration = max_duration
        self.is_recording = False
        self.temp_file = None
    
    def run(self):
        """녹음 시작"""
        self.is_recording = True
        
        # 임시 파일 생성
        fd, self.temp_file = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        # 녹음 설정
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        
        audio = pyaudio.PyAudio()
        
        # 녹음 스트림 열기
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                           rate=RATE, input=True,
                           frames_per_buffer=CHUNK)
        
        frames = []
        for i in range(0, int(RATE / CHUNK * self.max_duration)):
            if not self.is_recording:
                break
            
            data = stream.read(CHUNK)
            frames.append(data)
            
            # 진행 상황 업데이트 (0-100%)
            progress = int((i / (RATE / CHUNK * self.max_duration)) * 100)
            self.update_progress.emit(progress)
        
        # 스트림 닫기
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # WAV 파일로 저장
        wf = wave.open(self.temp_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # 완료 신호 보내기
        self.is_recording = False
        self.finished.emit(self.temp_file)
    
    def stop(self):
        """녹음 중지"""
        self.is_recording = False