"""
TranscriptionThread class - Handles audio transcription in a background thread.
"""

import wave
import numpy as np
import ctypes
import time
from PyQt6.QtCore import QThread, pyqtSignal

class TranscriptionThread(QThread):
   """음성 인식을 위한 스레드 클래스"""
   finished = pyqtSignal(str)
   progress = pyqtSignal(str)  # 실시간 텍스트 진행 상황 신호
   progress_percent = pyqtSignal(int)  # 진행률(%) 신호 추가
   error = pyqtSignal(str)
   
   def __init__(self, whisper, audio_file, language=None):
       super().__init__()
       self.whisper = whisper
       self.audio_file = audio_file
       self.language = language
       self.current_text = ""  # 현재까지의 인식 결과 저장
       self.total_segments = 0  # 예상 세그먼트 총 개수
       self.current_segment = 0  # 현재 처리한 세그먼트 수
   
   def run(self):
       """음성 인식 실행"""
       try:
           # 오디오 파일 로드
           wf = wave.open(self.audio_file, 'rb')
           
           # 오디오 파일 길이 계산 (초 단위)
           audio_length_sec = wf.getnframes() / wf.getframerate()
           
           # 예상 세그먼트 수 추정 (대략 5초당 1개 세그먼트로 가정)
           self.total_segments = max(1, int(audio_length_sec / 5))
           self.current_segment = 0
           
           # 오디오 데이터 읽기
           audio_data = np.frombuffer(
               wf.readframes(wf.getnframes()),
               dtype=np.int16
           ).astype(np.float32) / 32768.0  # 정규화
           
           # 초기 진행률 신호 발생
           self.progress_percent.emit(0)
           
           # 콜백 함수 정의
           @self.whisper.WHISPER_NEW_SEGMENT_CALLBACK
           def new_segment_callback(ctx, state, n_new, user_data):
               try:
                   # 새 세그먼트의 텍스트와 타임스탬프 가져오기
                   current_segments = self.whisper.get_current_segments()
                   
                   # 현재 세그먼트 수 증가
                   self.current_segment += n_new
                   
                   # 진행률 계산 및 신호 발생
                   progress = min(95, int((self.current_segment / self.total_segments) * 100))
                   self.progress_percent.emit(progress)
                   
                   # 실시간 텍스트 업데이트 신호 발생
                   self.progress.emit(current_segments)
               except Exception as e:
                   print(f"콜백 오류: {str(e)}")

           # 콜백과 함께 인식 수행
           text = self.whisper.transcribe(audio_data, self.language, new_segment_callback)
           
           # 완료 시 100% 진행률 설정
           self.progress_percent.emit(100)
           self.finished.emit(text)
       except Exception as e:
           self.error.emit(str(e))