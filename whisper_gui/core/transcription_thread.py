"""
TranscriptionThread class - Handles audio transcription in a background thread.
"""

import os
import wave
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class TranscriptionThread(QThread):
    """음성 인식을 위한 스레드 클래스"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, whisper, audio_file, language=None):
        super().__init__()
        self.whisper = whisper
        self.audio_file = audio_file
        self.language = language
    
    def run(self):
        """음성 인식 실행"""
        try:
            # 진행 상황 업데이트
            self.progress.emit(10)
            
            # 파일 확장자 확인
            file_ext = os.path.splitext(self.audio_file)[1].lower()
            
            # WAV 파일만 처리
            if file_ext != '.wav':
                raise Exception("현재 WAV 형식의 파일만 지원합니다.")
            
            # 진행 상황 업데이트
            self.progress.emit(20)
            
            # 오디오 파일 열기
            try:
                wf = wave.open(self.audio_file, 'rb')
                frames = wf.readframes(wf.getnframes())
                n_frames = wf.getnframes()
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                wf.close()
                
                # 진행 상황 업데이트
                self.progress.emit(40)
                
                # 오디오 데이터를 float32 배열로 변환
                if sample_width == 2:  # 16-bit audio
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 1:  # 8-bit audio
                    audio_data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                else:
                    raise Exception(f"지원되지 않는 오디오 형식입니다. (샘플 너비: {sample_width})")
                
                # 스테레오를 모노로 변환 (필요한 경우)
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                elif channels > 2:
                    raise Exception(f"지원되지 않는 채널 수입니다: {channels}")
                
                # 진행 상황 업데이트
                self.progress.emit(60)
                
                # 샘플레이트가 16kHz가 아닌 경우에 대한 처리가 필요할 수 있음
                # (현재 버전에서는 처리하지 않음)
                
                print(f"오디오 파일 정보: {n_frames} 프레임, {channels} 채널, {sample_rate} Hz, {sample_width} 바이트/샘플")
                
            except Exception as e:
                raise Exception(f"오디오 파일 로드 실패: {str(e)}")
            
            # 진행 상황 업데이트
            self.progress.emit(80)
            
            # 인식 수행
            try:
                text = self.whisper.transcribe(audio_data, self.language)
                self.progress.emit(100)
                self.finished.emit(text)
            except Exception as e:
                raise Exception(f"음성 인식 중 오류: {str(e)}")
        
        except Exception as e:
            self.error.emit(str(e))