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
        print(f"TranscriptionThread.__init__: audio_file={audio_file}, language={language}")
    
    def run(self):
        """음성 인식 실행"""
        try:
            print(f"TranscriptionThread.run: 시작")
            # 오디오 파일 로드
            print(f"오디오 파일 '{self.audio_file}' 열기")
            wf = wave.open(self.audio_file, 'rb')
            print(f"오디오 파일 정보: 채널 수: {wf.getnchannels()}, 샘플 너비: {wf.getsampwidth()}, 프레임 수: {wf.getnframes()}, 프레임 레이트: {wf.getframerate()}")
            
            # 오디오 데이터 읽기
            audio_data = np.frombuffer(
                wf.readframes(wf.getnframes()),
                dtype=np.int16
            ).astype(np.float32) / 32768.0  # 정규화
            
            print(f"오디오 데이터 로드 완료: 길이={len(audio_data)}, min={np.min(audio_data)}, max={np.max(audio_data)}")
            
            # 인식 수행
            print("whisper.transcribe 호출 전")
            try:
                text = self.whisper.transcribe(audio_data, self.language)
                print(f"whisper.transcribe 호출 후: 인식된 텍스트 = '{text}'")
                self.finished.emit(text)
            except Exception as e:
                print(f"whisper.transcribe 호출 중 예외 발생: {str(e)}")
                raise
        except Exception as e:
            print(f"TranscriptionThread.run 오류: {str(e)}")
            self.error.emit(str(e))