"""
개선된 WhisperGUI 클래스 - 메인 애플리케이션 창 및 사용자 인터페이스
"""

import os
import sys
from PyQt6.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QLabel, QComboBox, QTextEdit, 
                             QFileDialog, QProgressBar, QMessageBox, QApplication,
                             QTabWidget, QFrame, QSplitter)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon

from ..core.whisper_dll import WhisperDLL
from ..core.recording_thread import RecordingThread
from ..core.transcription_thread import TranscriptionThread
from ..core.utils import find_dll_file, check_required_dlls

from .device_selection_dialog import DeviceSelectionDialog
from .model_selection_dialog import ModelSelectionDialog

class WhisperGUI(QMainWindow):
    """Whisper 음성 인식 GUI 메인 클래스"""
    def __init__(self):
        super().__init__()
        
        # 윈도우 설정
        self.setWindowTitle("Whisper 다국어 음성 인식 프로그램")
        self.setGeometry(100, 100, 900, 600)
        
        # 모델 파일 경로 저장 변수
        self.model_file_path = None
        
        # Whisper DLL 인스턴스 관련 변수
        self.whisper = None
        self.vulkan_enabled = True  # 기본값: Vulkan 사용
        self.model_loaded = False
        
        # 스레드 초기화
        self.recording_thread = None
        self.transcription_thread = None
        
        # 실행 파일이 있는 디렉토리 경로
        self.current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 모델 디렉토리 생성 및 확인
        self.models_dir = os.path.join(self.current_dir, "models")
        if not os.path.exists(self.models_dir):
            try:
                os.makedirs(self.models_dir)
                print(f"모델 디렉토리 생성: {self.models_dir}")
            except Exception as e:
                print(f"모델 디렉토리 생성 실패: {str(e)}")
        else:
            print(f"모델 디렉토리 확인: {self.models_dir}")
        
        # UI 초기화
        self.init_ui()
        
        # 장치 선택 다이얼로그 표시
        QTimer.singleShot(100, self.show_device_selection)
    
    def init_ui(self):
        """UI 구성 요소 초기화"""
        # 메인 위젯 및 레이아웃
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 상단 정보 영역
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        info_frame.setLineWidth(1)
        info_layout = QHBoxLayout(info_frame)
        
        # 모델 정보
        model_layout = QVBoxLayout()
        model_title = QLabel("모델:")
        model_title.setFont(QFont("", 9, QFont.Weight.Bold))
        self.model_path_label = QLabel("모델이 로드되지 않음")
        self.model_path_label.setStyleSheet("color: red;")
        
        model_layout.addWidget(model_title)
        model_layout.addWidget(self.model_path_label)
        info_layout.addLayout(model_layout)
        
        # 가속 모드 정보
        accel_layout = QVBoxLayout()
        accel_title = QLabel("가속 모드:")
        accel_title.setFont(QFont("", 9, QFont.Weight.Bold))
        self.accel_label = QLabel("설정되지 않음")
        
        accel_layout.addWidget(accel_title)
        accel_layout.addWidget(self.accel_label)
        info_layout.addLayout(accel_layout)
        
        # 버튼
        btn_layout = QVBoxLayout()
        change_device_btn = QPushButton("장치 변경")
        change_device_btn.clicked.connect(self.show_device_selection)
        
        change_model_btn = QPushButton("모델 변경")
        change_model_btn.clicked.connect(self.show_model_selection)
        
        btn_layout.addWidget(change_device_btn)
        btn_layout.addWidget(change_model_btn)
        info_layout.addLayout(btn_layout)
        
        main_layout.addWidget(info_frame)
        
        # 입력 방식 탭
        self.tabs = QTabWidget()
        
        # 마이크 탭
        mic_tab = QWidget()
        mic_layout = QVBoxLayout(mic_tab)
        
        # 언어 선택 (마이크 탭)
        mic_lang_layout = QHBoxLayout()
        mic_lang_label = QLabel("인식 언어:")
        self.mic_language_combo = QComboBox()
        
        # 주요 언어 추가
        languages = [
            ("한국어", "ko"),
            ("영어", "en"),
            ("일본어", "ja"),
            ("중국어", "zh"),
            ("독일어", "de"),
            ("프랑스어", "fr"),
            ("스페인어", "es"),
            ("이탈리아어", "it"),
            ("러시아어", "ru")
        ]
        
        for name, code in languages:
            self.mic_language_combo.addItem(name, code)
        
        mic_lang_layout.addWidget(mic_lang_label)
        mic_lang_layout.addWidget(self.mic_language_combo)
        mic_layout.addLayout(mic_lang_layout)
        
        # 녹음 버튼
        self.record_btn = QPushButton("녹음 시작")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)  # 모델 로드 전에는 비활성화
        mic_layout.addWidget(self.record_btn)
        
        # 마이크 탭 진행 상황 바
        self.mic_progress_bar = QProgressBar()
        self.mic_progress_bar.setRange(0, 100)
        self.mic_progress_bar.setValue(0)
        self.mic_progress_bar.setFormat("%p%")
        mic_layout.addWidget(self.mic_progress_bar)
        
        self.tabs.addTab(mic_tab, "마이크 녹음")
        
        # 파일 탭
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        
        # 언어 선택 (파일 탭)
        file_lang_layout = QHBoxLayout()
        file_lang_label = QLabel("인식 언어:")
        self.file_language_combo = QComboBox()
        
        # 언어 목록 복제
        for name, code in languages:
            self.file_language_combo.addItem(name, code)
        
        file_lang_layout.addWidget(file_lang_label)
        file_lang_layout.addWidget(self.file_language_combo)
        file_layout.addLayout(file_lang_layout)
        
        # 파일 선택 버튼 및 경로 표시
        file_select_layout = QHBoxLayout()
        self.file_btn = QPushButton("오디오 파일 선택")
        self.file_btn.clicked.connect(self.select_audio_file)
        self.file_btn.setEnabled(False)  # 모델 로드 전에는 비활성화
        
        self.file_path_label = QLabel("파일이 선택되지 않음")
        file_select_layout.addWidget(self.file_btn)
        file_select_layout.addWidget(self.file_path_label, 1)
        
        file_layout.addLayout(file_select_layout)
        
        # 파일 탭 진행 상황 바
        self.file_progress_bar = QProgressBar()
        self.file_progress_bar.setRange(0, 100)
        self.file_progress_bar.setValue(0)
        self.file_progress_bar.setFormat("%p%")
        file_layout.addWidget(self.file_progress_bar)
        
        self.tabs.addTab(file_tab, "파일 업로드")
        
        main_layout.addWidget(self.tabs)
        
        # 결과 텍스트 영역
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.Shape.StyledPanel)
        result_layout = QVBoxLayout(result_frame)
        
        result_label = QLabel("인식 결과:")
        result_label.setFont(QFont("", 10, QFont.Weight.Bold))
        result_layout.addWidget(result_label)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        
        # 버튼 섹션
        button_layout = QHBoxLayout()
        
        clear_btn = QPushButton("지우기")
        clear_btn.clicked.connect(self.clear_results)
        
        copy_btn = QPushButton("복사")
        copy_btn.clicked.connect(self.copy_results)
        
        save_btn = QPushButton("저장")
        save_btn.clicked.connect(self.save_results)
        
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(copy_btn)
        button_layout.addWidget(save_btn)
        result_layout.addLayout(button_layout)
        
        main_layout.addWidget(result_frame)
        
        # 상태 표시줄
        self.statusBar().showMessage("시작하려면 장치와 모델을 선택하세요.")
    
    def show_device_selection(self):
        """장치 선택 다이얼로그 표시"""
        dialog = DeviceSelectionDialog(self)
        dialog.device_selected.connect(self.on_device_selected)
        
        # 윈도우 중앙에 표시
        dialog.move(self.frameGeometry().center() - dialog.rect().center())
        
        if dialog.exec() == 0:  # 취소 시 다음 단계로 넘어가지 않음
            if not self.model_file_path:  # 모델이 선택되지 않은 경우에만 메시지 표시
                self.statusBar().showMessage("시작하려면 장치와 모델을 선택하세요.")
    
    def on_device_selected(self, use_gpu):
        """장치 선택 결과 처리"""
        # 기존 모델이 메모리에 로드되어 있는 경우 먼저 해제
        if self.model_loaded and self.whisper:
            self.statusBar().showMessage("가속 모드 변경으로 인해 기존 모델을 메모리에서 해제하는 중...")
            self.unload_model()
            
        # Vulkan 설정 저장
        self.vulkan_enabled = use_gpu
        
        # UI 업데이트
        self.accel_label.setText("Vulkan GPU" if use_gpu else "CPU")
        self.accel_label.setStyleSheet("color: green;")
        
        # DLL 초기화 위치를 현재 위치로 변경 (초기화 시점을 실제 사용 직전으로 늦춤)
        self.whisper = None  # 기존 인스턴스 제거
        self.model_loaded = False
        
        # 모델 로드 지시
        self.statusBar().showMessage("장치가 선택되었습니다. 이제 모델을 선택하세요.")
        
        # 모델 선택 다이얼로그 표시
        self.show_model_selection()
    
    def show_model_selection(self):
        """모델 선택 다이얼로그 표시"""
        # 장치가 선택되지 않은 경우
        if not hasattr(self, 'vulkan_enabled'):
            self.show_device_selection()
            return
            
        dialog = ModelSelectionDialog(self.vulkan_enabled, self)
        
        # 모델 디렉토리 설정 - 일관성 유지
        dialog.models_dir = self.models_dir
        
        dialog.model_selected.connect(self.on_model_selected)
        
        # 윈도우 중앙에 표시
        dialog.move(self.frameGeometry().center() - dialog.rect().center())
        
        if dialog.exec() == 0:  # 취소 시
            if not self.model_file_path:  # 모델이 선택되지 않은 경우에만 메시지 표시
                self.statusBar().showMessage("시작하려면 모델을 선택하세요.")
    
    def on_model_selected(self, model_path):
        """모델 선택 결과 처리"""
        # 기존 모델이 메모리에 로드되어 있는 경우 먼저 해제
        if self.model_loaded and self.whisper and hasattr(self.whisper, 'ctx') and self.whisper.ctx:
            self.statusBar().showMessage("기존 모델을 메모리에서 해제하는 중...")
            self.unload_model()
            
        # 모델 경로 저장
        self.model_file_path = model_path
        
        # UI 업데이트
        self.model_path_label.setText(os.path.basename(model_path))
        self.model_path_label.setStyleSheet("color: green;")
        
        # 버튼 활성화
        self.record_btn.setEnabled(True)
        self.file_btn.setEnabled(True)
        
        # 상태 메시지 업데이트
        self.statusBar().showMessage("모델이 선택되었습니다. 음성을 녹음하거나 파일을 업로드하여 인식을 시작하세요.")
        
        # 모델은 아직 메모리에 로드하지 않음 (첫 사용 시 로드)
        self.model_loaded = False
    
    def toggle_recording(self):
        """녹음 시작/중지 전환"""
        if not self.model_file_path:
            QMessageBox.warning(self, "오류", "먼저 Whisper 모델 파일을 선택하세요.")
            return
        
        if self.recording_thread and self.recording_thread.is_recording:
            # 녹음 중지
            self.recording_thread.stop()
            self.record_btn.setText("녹음 시작")
            self.file_btn.setEnabled(True)
        else:
            # 녹음 시작
            self.record_btn.setText("녹음 중지")
            self.file_btn.setEnabled(False)
            self.mic_progress_bar.setValue(0)
            self.result_text.clear()
            
            # 녹음 스레드 생성 및 시작
            self.recording_thread = RecordingThread()
            self.recording_thread.update_progress.connect(self.update_mic_progress)
            self.recording_thread.finished.connect(self.on_recording_finished)
            self.recording_thread.start()
            
            # 상태 메시지 업데이트
            self.statusBar().showMessage("녹음 중... 중지 버튼을 클릭하여 종료하세요.")
    
    def update_mic_progress(self, value):
        """마이크 탭 진행 상황 업데이트"""
        self.mic_progress_bar.setValue(value)
    
    def on_recording_finished(self, audio_file):
        """녹음 완료 후 처리"""
        self.record_btn.setText("녹음 시작")
        self.file_btn.setEnabled(True)
        
        # 녹음된 파일로 인식 시작
        self.statusBar().showMessage("녹음이 완료되었습니다. 인식을 시작합니다...")
        self.transcribe_audio(audio_file, self.mic_language_combo.currentData(), self.mic_progress_bar)
    
    def select_audio_file(self):
        """오디오 파일 선택"""
        if not self.model_file_path:
            QMessageBox.warning(self, "오류", "먼저 Whisper 모델 파일을 선택하세요.")
            return
        
        file_dialog = QFileDialog()
        audio_file, _ = file_dialog.getOpenFileName(
            self, "오디오 파일 선택", "", "오디오 파일 (*.wav *.mp3 *.ogg);;모든 파일 (*.*)"
        )
        
        if audio_file:
            # UI 업데이트
            self.file_path_label.setText(os.path.basename(audio_file))
            self.result_text.clear()
            self.file_progress_bar.setValue(0)
            
            # 파일 인식 시작
            self.statusBar().showMessage(f"파일을 로드했습니다: {os.path.basename(audio_file)}. 인식을 시작합니다...")
            self.transcribe_audio(audio_file, self.file_language_combo.currentData(), self.file_progress_bar)
    
    def initialize_whisper(self):
        """Whisper DLL 인스턴스 초기화"""
        if self.whisper is not None:
            return True
        
        try:
            # DLL 파일 찾기
            possible_dll_names = ["whisper.dll", "whisper_cpp.dll", "whisper-cpp.dll", "whisperdll.dll"]
            dll_path = find_dll_file(self.current_dir, possible_dll_names)
            
            if not dll_path:
                raise Exception("Whisper DLL 파일을 찾을 수 없습니다.")
            
            # Whisper DLL 인스턴스 생성
            self.whisper = WhisperDLL(dll_path=dll_path, vulkan_support=self.vulkan_enabled)
            return True
        
        except Exception as e:
            QMessageBox.critical(self, "오류", f"Whisper DLL 초기화 실패: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_file, language, progress_bar):
        """오디오 파일 인식"""
        if not self.model_file_path:
            QMessageBox.warning(self, "오류", "먼저 Whisper 모델 파일을 선택하세요.")
            return
        
        # Whisper DLL 초기화 확인
        if not self.initialize_whisper():
            return
        
        # UI 업데이트
        progress_bar.setValue(0)
        self.result_text.clear()
        self.result_text.setPlaceholderText("모델 로드 중...")
        
        # 이 시점에서 모델을 실제로 로드 (필요한 경우)
        if not self.model_loaded or not hasattr(self.whisper, 'ctx') or not self.whisper.ctx:
            try:
                # 기존 모델이 있으면 먼저 해제
                if self.model_loaded:
                    self.unload_model()
                
                progress_bar.setRange(0, 0)  # 불확정 진행 상황 모드
                QApplication.processEvents()  # UI 업데이트
                
                # 모델 로드
                self.statusBar().showMessage("모델을 메모리에 로드하는 중... 잠시만 기다려주세요.")
                self.whisper.load_model(self.model_file_path)
                self.model_loaded = True
                
                # 로드 성공 표시
                self.model_path_label.setText(f"{os.path.basename(self.model_file_path)} ({self.whisper.acceleration_mode} 모드) - 로드됨")
                progress_bar.setRange(0, 100)  # 범위 복원
            except Exception as e:
                progress_bar.setRange(0, 100)
                progress_bar.setValue(0)
                self.result_text.setPlaceholderText("")
                QMessageBox.critical(self, "모델 로드 오류", str(e))
                return
        
        # 이제 로드된 모델로 인식 시작
        self.result_text.setPlaceholderText("인식 중...")
        self.statusBar().showMessage("음성을 텍스트로 변환하는 중...")
        
        # 파일 인식 시작
        self.transcription_thread = TranscriptionThread(self.whisper, audio_file, language)
        self.transcription_thread.finished.connect(self.on_transcription_finished)
        self.transcription_thread.progress.connect(self.on_transcription_progress)  # 실시간 텍스트 업데이트
        self.transcription_thread.progress_percent.connect(lambda value: progress_bar.setValue(value))  # 진행률 업데이트
        self.transcription_thread.error.connect(self.on_transcription_error)
        self.transcription_thread.start()
    
    def on_transcription_progress(self, text):
        """인식 진행 상황 텍스트 업데이트 (실시간)"""
        self.result_text.setPlaceholderText("")
        self.result_text.setText(text)

    def on_transcription_finished(self, text):
        """인식 완료 후 처리"""
        active_tab = self.tabs.currentIndex()
        if active_tab == 0:  # 마이크 탭
            self.mic_progress_bar.setValue(100)
        else:  # 파일 탭
            self.file_progress_bar.setValue(100)
            
        self.result_text.setPlaceholderText("")
        self.result_text.setText(text)
        
        self.statusBar().showMessage("음성 인식이 완료되었습니다.")
        
    def on_transcription_error(self, error_msg):
        """인식 오류 처리"""
        active_tab = self.tabs.currentIndex()
        if active_tab == 0:  # 마이크 탭
            self.mic_progress_bar.setValue(0)
        else:  # 파일 탭
            self.file_progress_bar.setValue(0)
            
        self.result_text.setPlaceholderText("")
        QMessageBox.critical(self, "인식 오류", error_msg)
        
        self.unload_model()
        self.statusBar().showMessage("오류가 발생했습니다. 다시 시도하세요.")
        
    def unload_model(self):
        """모델을 메모리에서 해제"""
        try:
            if self.whisper and hasattr(self.whisper, 'ctx') and self.whisper.ctx:
                print("모델 메모리 해제 시작")
                # 새 free_model 메서드 사용
                result = self.whisper.free_model()
                self.model_loaded = False
                
                # UI 업데이트
                if self.model_file_path:
                    self.model_path_label.setText(f"{os.path.basename(self.model_file_path)} - 준비됨 (메모리 해제됨)")
                    
                print(f"모델 메모리 해제 완료: {result}")
                return result
            return True
        except Exception as e:
            print(f"모델 해제 중 오류 발생: {str(e)}")
            return False
        
    def clear_results(self):
        """결과 지우기"""
        self.result_text.clear()
        self.mic_progress_bar.setValue(0)
        self.file_progress_bar.setValue(0)
        self.statusBar().showMessage("텍스트가 지워졌습니다.")
    
    def copy_results(self):
        """결과 클립보드에 복사"""
        text = self.result_text.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.statusBar().showMessage("텍스트가 클립보드에 복사되었습니다.")
        else:
            self.statusBar().showMessage("복사할 텍스트가 없습니다.")
    
    def save_results(self):
        """결과 파일로 저장"""
        text = self.result_text.toPlainText()
        if not text:
            QMessageBox.warning(self, "저장 실패", "저장할 텍스트가 없습니다.")
            return
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "결과 저장", "", "텍스트 파일 (*.txt);;모든 파일 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.statusBar().showMessage(f"결과가 {file_path}에 저장되었습니다.")
            except Exception as e:
                QMessageBox.critical(self, "저장 실패", f"파일 저장 중 오류 발생: {str(e)}")
    
    def closeEvent(self, event):
        """프로그램 종료 시 처리"""
        try:
            # 녹음 중이면 중지
            if self.recording_thread and self.recording_thread.is_recording:
                self.recording_thread.stop()
            
            # 임시 파일 삭제
            if self.recording_thread and hasattr(self.recording_thread, 'temp_file') and self.recording_thread.temp_file:
                try:
                    if os.path.exists(self.recording_thread.temp_file):
                        os.remove(self.recording_thread.temp_file)
                except Exception as e:
                    print(f"임시 파일 삭제 실패: {str(e)}")
            
            # Whisper 리소스 해제
            self.unload_model()
            
            print("프로그램 종료")
        except Exception as e:
            print(f"프로그램 종료 중 오류 발생: {str(e)}")
        
        event.accept()