"""
TranscriptionThread class - Handles audio transcription in a background thread.
"""

import wave
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class TranscriptionThread(QThread):
   """음성 인식을 위한 스레드 클래스"""
   finished = pyqtSignal(str)
   error = pyqtSignal(str)
   
   def __init__(self, whisper, audio_file, language=None):
       super().__init__()
       self.whisper = whisper
       self.audio_file = audio_file
       self.language = language
   
   def run(self):
       """음성 인식 실행"""
       try:
           # 오디오 파일 로드
           wf = wave.open(self.audio_file, 'rb')
           
           # 오디오 데이터 읽기
           audio_data = np.frombuffer(
               wf.readframes(wf.getnframes()),
               dtype=np.int16
           ).astype(np.float32) / 32768.0  # 정규화
           
           # 인식 수행
           text = self.whisper.transcribe(audio_data, self.language)
           self.finished.emit(text)
       except Exception as e:
           self.error.emit(str(e))