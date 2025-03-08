"""
모델 선택 다이얼로그 - 기존 모델 선택 또는 새 모델 다운로드
"""

import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QListWidget, QListWidgetItem,
                             QMessageBox, QFileDialog, QProgressBar, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon

from ..core.model_downloader import ModelDownloader

class ModelSelectionDialog(QDialog):
    """사용자가 Whisper 모델을 선택하거나 다운로드할 수 있는 다이얼로그"""
    
    # 모델 선택 완료 시 신호 전송
    model_selected = pyqtSignal(str)  # 선택된 모델 파일 경로
    
    def __init__(self, vulkan_support=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Whisper - 모델 선택")
        self.setMinimumSize(600, 400)
        self.setModal(True)
        
        self.vulkan_support = vulkan_support
        
        # 모델 디렉토리
        self.current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = os.path.join(self.current_dir, "models")
        
        # 부모 객체에서 모델 디렉토리 가져오기 (일관성 유지)
        if parent and hasattr(parent, 'models_dir'):
            self.models_dir = parent.models_dir
        
        # 모델 디렉토리 생성 (없는 경우)
        if not os.path.exists(self.models_dir):
            try:
                os.makedirs(self.models_dir)
            except Exception as e:
                print(f"모델 디렉토리 생성 실패: {str(e)}")
        
        # UI 초기화
        self.init_ui()
        
        # 모델 목록 로드
        self.load_models()
        
    def init_ui(self):
        """UI 구성"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # 타이틀 레이블
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        
        title_label = QLabel("Whisper 음성 인식 모델 선택")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 가속 모드 표시
        accel_label = QLabel(f"선택된 가속 모드: {'Vulkan GPU' if self.vulkan_support else 'CPU'}")
        accel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(accel_label)
        
        # 모델 목록 영역
        model_frame = QFrame()
        model_frame.setFrameShape(QFrame.Shape.StyledPanel)
        model_frame.setLineWidth(1)
        
        model_layout = QVBoxLayout(model_frame)
        
        model_header = QLabel("설치된 모델 목록")
        model_header.setFont(QFont("", 10, QFont.Weight.Bold))
        model_layout.addWidget(model_header)
        
        # 모델 목록
        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(150)
        self.model_list.itemDoubleClicked.connect(self.on_model_double_clicked)
        model_layout.addWidget(self.model_list)
        
        # 모델 관리 버튼들
        btn_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.load_models)
        
        self.download_btn = QPushButton("모델 다운로드")
        self.download_btn.clicked.connect(self.show_model_downloader)
        
        self.browse_btn = QPushButton("모델 파일 찾기")
        self.browse_btn.clicked.connect(self.browse_model_file)
        
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.download_btn)
        btn_layout.addWidget(self.browse_btn)
        
        model_layout.addLayout(btn_layout)
        layout.addWidget(model_frame)
        
        # 상태 메시지
        self.status_label = QLabel("모델을 선택하거나 다운로드하세요.")
        layout.addWidget(self.status_label)
        
        # 하단 버튼
        bottom_btn_layout = QHBoxLayout()
        
        back_btn = QPushButton("이전")
        back_btn.clicked.connect(self.reject)
        
        self.next_btn = QPushButton("다음")
        self.next_btn.clicked.connect(self.on_next_clicked)
        self.next_btn.setEnabled(False)
        self.next_btn.setDefault(True)
        
        bottom_btn_layout.addWidget(back_btn)
        bottom_btn_layout.addWidget(self.next_btn)
        
        layout.addLayout(bottom_btn_layout)
        self.setLayout(layout)
    
    def load_models(self):
        """설치된 모델 파일 목록 로드"""
        self.model_list.clear()
        self.selected_model_path = None
        self.next_btn.setEnabled(False)
        
        # models 디렉토리 검사
        if not os.path.exists(self.models_dir):
            self.status_label.setText("모델 디렉토리가 없습니다. 모델을 다운로드하세요.")
            return
        
        # 모델 파일 목록 가져오기
        model_files = []
        for file in os.listdir(self.models_dir):
            if file.endswith('.bin') or 'ggml' in file:
                model_files.append(file)
        
        if not model_files:
            self.status_label.setText("설치된 모델이 없습니다. 모델을 다운로드하세요.")
            return
        
        # 모델 파일 목록에 추가
        for file in sorted(model_files):
            file_path = os.path.join(self.models_dir, file)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # 모델 정보 파싱 (ggml-small.bin -> small)
            model_name = file.replace('ggml-', '').replace('.bin', '')
            
            # 언어 정보 및 크기 (tiny.en -> 영어 전용 tiny)
            lang_info = "다국어"
            if '.en' in model_name:
                lang_info = "영어 전용"
                model_name = model_name.replace('.en', '')
            
            # 아이템 생성
            item = QListWidgetItem(f"{model_name} ({lang_info}, {file_size_mb:.1f} MB)")
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            self.model_list.addItem(item)
        
        self.status_label.setText(f"{len(model_files)}개의 모델이 설치되어 있습니다.")
    
    def show_model_downloader(self):
        """모델 다운로드 다이얼로그 표시"""
        # ModelDownloader에 현재 모델 디렉토리 전달
        downloader = ModelDownloader(self)
        # 모델 디렉토리 설정
        downloader.models_dir = self.models_dir
        downloader.path_label.setText(self.models_dir)
        downloader.exec()  # 모달 대화상자로 실행
        
        # 다운로드 후 모델 리스트 갱신
        self.load_models()
    
    def browse_model_file(self):
        """파일 선택 대화상자로 모델 파일 찾기"""
        file_dialog = QFileDialog()
        # 기본 시작 디렉토리를 모델 폴더로 설정
        start_dir = self.models_dir if os.path.exists(self.models_dir) else ""
        
        model_path, _ = file_dialog.getOpenFileName(
            self, "Whisper 모델 파일 선택", start_dir, 
            "모델 파일 (*.bin *.ggml);;모든 파일 (*.*)"
        )
        
        if model_path:
            # 파일 크기 검증 (최소 크기 검사)
            file_size = os.path.getsize(model_path)
            if file_size < 1024:  # 1KB 미만은 유효한 모델 파일이 아닐 가능성이 높음
                QMessageBox.warning(self, "경고", f"선택한 파일이 너무 작습니다 ({file_size} bytes). 올바른 모델 파일이 아닐 수 있습니다.")
                return
            
            # 모델 파일 선택 처리
            self.selected_model_path = model_path
            self.status_label.setText(f"선택된 모델: {os.path.basename(model_path)}")
            self.next_btn.setEnabled(True)
    
    def on_model_double_clicked(self, item):
        """리스트에서 모델 더블 클릭 처리"""
        model_path = item.data(Qt.ItemDataRole.UserRole)
        if model_path:
            self.selected_model_path = model_path
            self.status_label.setText(f"선택된 모델: {os.path.basename(model_path)}")
            self.next_btn.setEnabled(True)
    
    def on_next_clicked(self):
        """다음 버튼 클릭 처리"""
        if not self.selected_model_path:
            # 목록에서 선택된 항목 확인
            selected_items = self.model_list.selectedItems()
            if selected_items:
                self.selected_model_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
                print(f"선택된 모델 경로: {self.selected_model_path}")
            else:
                QMessageBox.warning(self, "모델 선택 필요", "계속하려면 모델을 선택하세요.")
                return
        
        # 선택된 모델 신호 전송
        self.model_selected.emit(self.selected_model_path)
        self.accept()
    
    def on_model_downloaded(self, model_path):
        """모델이 다운로드된 후 호출되는 메서드"""
        if not model_path or not os.path.exists(model_path):
            return
        
        # 새로 다운로드된 모델을 현재 모델로 설정
        self.selected_model_path = model_path
        self.status_label.setText(f"선택된 모델: {os.path.basename(model_path)}")
        self.next_btn.setEnabled(True)
        
        # 모델 목록 갱신
        self.load_models()
        
        # 다운로드된 모델 선택
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == model_path:
                self.model_list.setCurrentItem(item)
                break