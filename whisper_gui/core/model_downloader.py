"""
모델 다운로더 모듈 - 사용자가 Whisper 모델을 편리하게 다운로드할 수 있는 기능 제공
"""

import os
import sys
import subprocess
import threading
import requests
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QComboBox, QPushButton, QProgressBar, 
                              QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# 다운로드 가능한 모델 목록
AVAILABLE_MODELS = [
    ("tiny.en", "영어 전용 - tiny (약 75MB)"),
    ("tiny", "다국어 - tiny (약 75MB)"),
    ("base.en", "영어 전용 - base (약 142MB)"),
    ("base", "다국어 - base (약 142MB)"),
    ("small.en", "영어 전용 - small (약 466MB)"),
    ("small", "다국어 - small (약 466MB)"),
    ("medium.en", "영어 전용 - medium (약 1.5GB)"),
    ("medium", "다국어 - medium (약 1.5GB)"),
    ("large-v1", "다국어 - large v1 (약 2.9GB)"),
    ("large-v2", "다국어 - large v2 (약 2.9GB)"),
    ("large-v3", "다국어 - large v3 (약 2.9GB)"),
    ("large-v3-turbo", "다국어 - large v3 Turbo (약 1.5GB)"),
]

# 모델 다운로드 관련 신호를 전달하기 위한 클래스
class DownloadSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    status = pyqtSignal(str)

class ModelDownloader(QDialog):
    """Whisper 모델 다운로드 다이얼로그"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Whisper 모델 다운로드")
        self.setGeometry(100, 100, 500, 200)
        self.setModal(True)
        
        # 신호 객체 생성
        self.signals = DownloadSignals()
        
        # 앱 루트 디렉토리 (모델 저장 위치)
        self.app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.app_root, "models")
        
        # 부모에게서 models_dir 상속 (일관성 유지)
        if parent and hasattr(parent, 'models_dir'):
            self.models_dir = parent.models_dir
            print(f"부모로부터 모델 디렉토리 경로 상속: {self.models_dir}")
        else:
            print(f"기본 모델 디렉토리 사용: {self.models_dir}")
        
        # 다운로드 중인지 여부
        self.is_downloading = False
        self.download_thread = None
        
        # UI 초기화
        self.init_ui()
        
    def init_ui(self):
        """UI 구성"""
        layout = QVBoxLayout()
        
        # 모델 선택
        model_layout = QHBoxLayout()
        model_label = QLabel("모델 선택:")
        self.model_combo = QComboBox()
        
        # 모델 목록 추가
        for model_code, model_desc in AVAILABLE_MODELS:
            self.model_combo.addItem(f"{model_desc}", model_code)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1)
        layout.addLayout(model_layout)
        
        # 저장 경로
        path_layout = QHBoxLayout()
        path_label = QLabel("저장 경로:")
        self.path_label = QLabel(self.models_dir)
        self.browse_btn = QPushButton("찾아보기")
        self.browse_btn.clicked.connect(self.browse_path)
        
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_label, 1)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)
        
        # 상태 표시
        self.status_label = QLabel("모델을 선택하고 다운로드 버튼을 클릭하세요.")
        layout.addWidget(self.status_label)
        
        # 진행 상황 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")  # 퍼센트 표시 포맷 설정
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignRight)  # 텍스트 가운데 정렬
        layout.addWidget(self.progress_bar)
        
        # 버튼
        button_layout = QHBoxLayout()
        self.download_btn = QPushButton("다운로드")
        self.download_btn.clicked.connect(self.start_download)
        
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.download_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # 신호 연결
        self.signals.progress.connect(self.update_progress)
        self.signals.finished.connect(self.download_finished)
        self.signals.status.connect(self.update_status)
    
    def browse_path(self):
        """저장 경로 선택 대화상자"""
        directory = QFileDialog.getExistingDirectory(self, "저장 경로 선택", self.models_dir)
        if directory:
            self.models_dir = directory
            self.path_label.setText(directory)
    
    def start_download(self):
        """모델 다운로드 시작"""
        if self.is_downloading:
            return
        
        # 모델 정보 가져오기
        model_code = self.model_combo.currentData()
        model_desc = self.model_combo.currentText()
        
        # 저장 경로 확인
        if not os.path.exists(self.models_dir):
            try:
                os.makedirs(self.models_dir)
            except Exception as e:
                QMessageBox.critical(self, "오류", f"저장 경로를 생성할 수 없습니다: {str(e)}")
                return
        
        # 파일명 생성
        model_filename = f"ggml-{model_code}.bin"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # 파일이 이미 존재하는지 확인
        if os.path.exists(model_path):
            reply = QMessageBox.question(
                self, "파일 존재", 
                f"모델 파일이 이미 존재합니다. 덮어쓰시겠습니까?\n{model_path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # UI 업데이트
        self.is_downloading = True
        self.download_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.cancel_btn.setText("다운로드 취소")
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.cancel_download)
        
        # 다운로드 시작 (스레드에서 실행)
        self.signals.status.emit(f"{model_desc} 다운로드 준비 중...")
        self.download_thread = threading.Thread(
            target=self.download_model,
            args=(model_code, model_path)
        )
        self.download_thread.daemon = True
        self.download_thread.start()
    
    def download_model(self, model_code, model_path):
        """모델 다운로드 스레드"""
        try:
            # Hugging Face URL 생성
            url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_code}.bin"
            self.signals.status.emit(f"다운로드 중: {url}")
            
            # Requests로 다운로드 (Progress 추적 기능 포함)
            response = requests.get(url, stream=True)
            response.raise_for_status()  # HTTP 오류 확인
            
            # 파일 크기 계산
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB 단위로 다운로드
            
            # 파일 다운로드 및 진행 상황 업데이트
            downloaded = 0
            with open(model_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    if not self.is_downloading:  # 취소 확인
                        raise Exception("다운로드가 취소되었습니다.")
                    
                    # 데이터 쓰기
                    f.write(data)
                    downloaded += len(data)
                    
                    # 진행 상황 계산 및 업데이트
                    if total_size > 0:
                        progress = int((downloaded / total_size) * 100)
                        self.signals.progress.emit(progress)
                        
                        # 다운로드 크기 표시
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        self.signals.status.emit(f"다운로드 중: {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({progress}%)")
            
            # 다운로드 완료 신호 전송
            self.signals.finished.emit(True, model_path)
        
        except Exception as e:
            # 오류 발생 시 파일 삭제 시도
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
            except:
                pass
            
            # 오류 신호 전송
            self.signals.finished.emit(False, str(e))
    
    def cancel_download(self):
        """다운로드 취소"""
        if self.is_downloading:
            self.is_downloading = False
            self.signals.status.emit("다운로드 취소 중...")
            # 스레드는 is_downloading 플래그를 확인하고 종료됨
    
    def update_progress(self, value):
        """진행 상황 바 업데이트"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """상태 메시지 업데이트"""
        self.status_label.setText(message)
    
    def download_finished(self, success, message):
        """다운로드 완료 처리"""
        self.is_downloading = False
        self.download_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.cancel_btn.setText("닫기")
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.close)
        
        if success:
            self.signals.status.emit(f"다운로드 완료: {message}")
            QMessageBox.information(self, "다운로드 성공", f"모델이 성공적으로 다운로드되었습니다.\n{message}")
            
            # 부모 창에 다운로드된 모델 정보 전달
            if self.parent():
                self.parent().on_model_downloaded(message)
            
            # 다이얼로그 닫기
            self.accept()
        else:
            self.signals.status.emit(f"다운로드 실패: {message}")
            QMessageBox.critical(self, "다운로드 실패", f"모델 다운로드에 실패했습니다.\n오류: {message}")
    
    def closeEvent(self, event):
        """다이얼로그가 닫힐 때 호출"""
        if self.is_downloading:
            reply = QMessageBox.question(
                self, "다운로드 중단", 
                "다운로드가 진행 중입니다. 중단하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.cancel_download()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()