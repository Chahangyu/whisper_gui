"""
TranscriptionThread class - Handles audio transcription in a background thread.
"""

import wave
import numpy as np
import ctypes
from PyQt6.QtCore import QThread, pyqtSignal

class TranscriptionThread(QThread):
   """음성 인식을 위한 스레드 클래스"""
   finished = pyqtSignal(str)
   progress = pyqtSignal(str)  # 실시간 진행 상황 신호 추가
   error = pyqtSignal(str)
   
   def __init__(self, whisper, audio_file, language=None):
       super().__init__()
       self.whisper = whisper
       self.audio_file = audio_file
       self.language = language
       self.current_text = ""  # 현재까지의 인식 결과 저장
   
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
           
           # 콜백 함수 정의
           @self.whisper.WHISPER_NEW_SEGMENT_CALLBACK
           def new_segment_callback(ctx, state, n_new, user_data):
               try:
                   # 새 세그먼트의 텍스트와 타임스탬프 가져오기
                   current_segments = self.whisper.get_current_segments()
                   # 실시간 업데이트 신호 발생
                   self.progress.emit(current_segments)
               except Exception as e:
                   print(f"콜백 오류: {str(e)}")

           # 콜백과 함께 인식 수행
           text = self.whisper.transcribe(audio_data, self.language, new_segment_callback)
           self.finished.emit(text)
       except Exception as e:
           self.error.emit(str(e))