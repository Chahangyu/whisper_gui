"""
WhisperDLL class - Interface for the Whisper speech recognition DLL
"""

import os
import ctypes
import numpy as np

# ggml.h 및 whisper.h에서 정의된 상수와 타입
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_CHUNK_SIZE = 30

# ggml 관련 타입 정의
class GgmlContext(ctypes.Structure):
    pass

class GgmlTensor(ctypes.Structure):
    pass

# whisper 관련 타입 정의
class WhisperContext(ctypes.Structure):
    pass

class WhisperState(ctypes.Structure):
    pass

class WhisperToken(ctypes.c_int32):
    pass

class WhisperTokenData(ctypes.Structure):
    _fields_ = [
        ("id", WhisperToken),        # 토큰 ID
        ("tid", WhisperToken),       # 강제 타임스탬프 토큰 ID
        ("p", ctypes.c_float),       # 토큰 확률
        ("plog", ctypes.c_float),    # 토큰 로그 확률
        ("pt", ctypes.c_float),      # 타임스탬프 토큰 확률
        ("ptsum", ctypes.c_float),   # 모든 타임스탬프 토큰의 확률 합계
        ("t0", ctypes.c_int64),      # 토큰 시작 시간
        ("t1", ctypes.c_int64),      # 토큰 종료 시간
        ("t_dtw", ctypes.c_int64),   # DTW를 사용한 토큰 시간
        ("vlen", ctypes.c_float),    # 토큰의 보이스 길이
    ]

# 콜백 함수 타입 정의
WHISPER_NEW_SEGMENT_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.POINTER(WhisperContext), ctypes.POINTER(WhisperState), ctypes.c_int, ctypes.c_void_p)
WHISPER_PROGRESS_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.POINTER(WhisperContext), ctypes.POINTER(WhisperState), ctypes.c_int, ctypes.c_void_p)
WHISPER_ENCODER_BEGIN_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(WhisperContext), ctypes.POINTER(WhisperState), ctypes.c_void_p)
GGML_ABORT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)
WHISPER_LOGITS_FILTER_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.POINTER(WhisperContext), ctypes.POINTER(WhisperState), ctypes.POINTER(WhisperTokenData), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p)

# 샘플링 전략 열거형
class WhisperSamplingStrategy(ctypes.c_int):
    WHISPER_SAMPLING_GREEDY = 0
    WHISPER_SAMPLING_BEAM_SEARCH = 1

# 정렬 문제를 방지하기 위한 greedy 및 beam_search 구조체
class WhisperGreedy(ctypes.Structure):
    _fields_ = [
        ("best_of", ctypes.c_int),   # 후보 샘플링 수
    ]

class WhisperBeamSearch(ctypes.Structure):
    _fields_ = [
        ("beam_size", ctypes.c_int), # 빔 크기
        ("patience", ctypes.c_float), # 패티언스
    ]

# whisper_full_params 구조체 정의 (whisper.h 기반)
class WhisperFullParams(ctypes.Structure):
    _pack_ = 8  # 메모리 정렬 지정
    _fields_ = [
        ("strategy", ctypes.c_int),                      # 샘플링 전략
        ("n_threads", ctypes.c_int),                     # 스레드 수
        ("n_max_text_ctx", ctypes.c_int),                # 최대 텍스트 컨텍스트
        ("offset_ms", ctypes.c_int),                     # 시작 오프셋 (ms)
        ("duration_ms", ctypes.c_int),                   # 처리할 오디오 길이 (ms)
        
        ("translate", ctypes.c_bool),                    # 번역 모드
        ("no_context", ctypes.c_bool),                   # 이전 컨텍스트 무시
        ("no_timestamps", ctypes.c_bool),                # 타임스탬프 생성 안함
        ("single_segment", ctypes.c_bool),               # 단일 세그먼트 출력
        ("print_special", ctypes.c_bool),                # 특수 토큰 출력
        ("print_progress", ctypes.c_bool),               # 진행 상황 출력
        ("print_realtime", ctypes.c_bool),               # 실시간 결과 출력
        ("print_timestamps", ctypes.c_bool),             # 타임스탬프 출력
        
        # 토큰 레벨 타임스탬프 (실험적)
        ("token_timestamps", ctypes.c_bool),             # 토큰별 타임스탬프 활성화
        ("thold_pt", ctypes.c_float),                    # 타임스탬프 토큰 확률 임계값
        ("thold_ptsum", ctypes.c_float),                 # 타임스탬프 토큰 합계 확률 임계값
        ("max_len", ctypes.c_int),                       # 최대 세그먼트 길이
        ("split_on_word", ctypes.c_bool),                # 단어 단위 분할
        ("max_tokens", ctypes.c_int),                    # 세그먼트당 최대 토큰 수
        
        # 속도 향상 기법 (실험적)
        ("debug_mode", ctypes.c_bool),                   # 디버그 모드 활성화
        ("audio_ctx", ctypes.c_int),                     # 오디오 컨텍스트 크기 설정
        
        # 화자 인식 (실험적)
        ("tdrz_enable", ctypes.c_bool),                  # 화자 전환 감지 활성화
        
        # 정규식
        ("suppress_regex", ctypes.c_char_p),             # 억제할 토큰 정규식
        
        # 초기 프롬프트
        ("initial_prompt", ctypes.c_char_p),             # 초기 프롬프트 텍스트
        ("prompt_tokens", ctypes.c_void_p),              # 프롬프트 토큰 배열
        ("prompt_n_tokens", ctypes.c_int),               # 프롬프트 토큰 수
        
        # 언어 설정
        ("language", ctypes.c_char_p),                   # 언어 코드
        ("detect_language", ctypes.c_bool),              # 언어 자동 감지
        
        # 공통 디코딩 파라미터
        ("suppress_blank", ctypes.c_bool),               # 빈 줄 억제
        ("suppress_nst", ctypes.c_bool),                 # 비음성 토큰 억제
        
        ("temperature", ctypes.c_float),                 # 초기 디코딩 온도
        ("max_initial_ts", ctypes.c_float),              # 최대 초기 타임스탬프
        ("length_penalty", ctypes.c_float),              # 길이 페널티
        
        # 폴백 파라미터
        ("temperature_inc", ctypes.c_float),             # 온도 증가
        ("entropy_thold", ctypes.c_float),               # 엔트로피 임계값
        ("logprob_thold", ctypes.c_float),               # 로그 확률 임계값
        ("no_speech_thold", ctypes.c_float),             # 무음 임계값
        
        # Greedy 탐색 파라미터
        ("greedy", WhisperGreedy),                      # greedy 파라미터
        
        # 빔 탐색 파라미터
        ("beam_search", WhisperBeamSearch),             # beam_search 파라미터
        
        # 콜백 함수들
        ("new_segment_callback", WHISPER_NEW_SEGMENT_CALLBACK),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        
        ("progress_callback", WHISPER_PROGRESS_CALLBACK),
        ("progress_callback_user_data", ctypes.c_void_p),
        
        ("encoder_begin_callback", WHISPER_ENCODER_BEGIN_CALLBACK),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        
        ("abort_callback", GGML_ABORT_CALLBACK),
        ("abort_callback_user_data", ctypes.c_void_p),
        
        ("logits_filter_callback", WHISPER_LOGITS_FILTER_CALLBACK),
        ("logits_filter_callback_user_data", ctypes.c_void_p),
        
        # 문법 규칙
        ("grammar_rules", ctypes.c_void_p),
        ("n_grammar_rules", ctypes.c_size_t),
        ("i_start_rule", ctypes.c_size_t),
        ("grammar_penalty", ctypes.c_float),
    ]

class WhisperDLL:
    def __init__(self, dll_path=None, vulkan_support=True):
        """Whisper DLL을 로드하고 필요한 함수를 설정합니다."""
        try:
            # GGML 관련 DLL 파일들 목록
            self.vulkan_support = vulkan_support
            self.extra_dlls = []
            self.acceleration_mode = "CPU"  # 기본값
            self.ctx = None  # 모델 컨텍스트 초기화
            self.current_callback = None  # 콜백 참조 저장 (가비지 컬렉션 방지)
            
            # 현재 디렉토리 경로
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up two levels to get to the app root directory
            current_dir = os.path.dirname(os.path.dirname(current_dir))
            
            # whisper.dll 경로 설정
            if dll_path is None:
                dll_path = os.path.join(current_dir, "whisper.dll")
                if not os.path.exists(dll_path):
                    # 현재 디렉토리에서 whisper.dll을 찾지 못한 경우, 다른 이름 시도
                    possible_names = ["whisper.dll", "whisper_cpp.dll", "whisper-cpp.dll", "whisperdll.dll"]
                    found = False
                    for name in possible_names:
                        test_path = os.path.join(current_dir, name)
                        if os.path.exists(test_path):
                            dll_path = test_path
                            found = True
                            break
                    
                    if not found:
                        raise Exception(f"whisper.dll 파일을 찾을 수 없습니다. '{current_dir}' 디렉토리에 파일이 있는지 확인하세요.")
            
            # 필요한 DLL 파일 목록
            ggml_dlls = [
                "ggml.dll",
                "ggml-base.dll",
                "ggml-cpu.dll",
            ]
            
            # Vulkan 지원이 요청된 경우 목록에 추가
            if vulkan_support:
                ggml_dlls.append("ggml-vulkan.dll")
            
            # 모든 필요한 DLL 파일을 로드
            for dll_name in ggml_dlls:
                dll_path_full = os.path.join(current_dir, dll_name)
                if os.path.exists(dll_path_full):
                    try:
                        ggml_dll = ctypes.CDLL(dll_path_full)
                        self.extra_dlls.append(ggml_dll)
                        
                        # 가속 모드 설정
                        if dll_name == "ggml-vulkan.dll" and vulkan_support:
                            self.acceleration_mode = "Vulkan GPU"
                    except Exception:
                        pass
            
            # 메인 Whisper DLL 로드
            self.dll = ctypes.CDLL(dll_path)
            
            # 콜백 타입 가져오기
            self.WHISPER_NEW_SEGMENT_CALLBACK = WHISPER_NEW_SEGMENT_CALLBACK
            
            # 함수 초기화
            if not self.initialize():
                raise Exception("DLL 함수 초기화 실패")
                
        except Exception as e:
            raise Exception(f"WhisperDLL 초기화 실패: {str(e)}")
    
    def load_model(self, model_path):
        """Whisper 모델을 로드합니다."""
        # 이미 모델이 로드되어 있는 경우 먼저 해제
        self.free_model()
        
        try:
            # 파일이 존재하는지 확인
            if not os.path.exists(model_path):
                raise Exception(f"모델 파일이 존재하지 않습니다: {model_path}")
            
            # 파일 경로를 바이트로 인코딩
            model_path_bytes = model_path.encode('utf-8')
            
            # 모델 로드 시도 (whisper_init_from_file 사용)
            self.ctx = self.dll.whisper_init_from_file(model_path_bytes)
            
            # 로드 실패 시
            if not self.ctx or self.ctx == 0:
                raise Exception("모델 초기화 실패: whisper_init_from_file이 null을 반환했습니다")
            
            return True
        except Exception as e:
            self.free_model()  # 오류 발생 시 모델 해제
            raise Exception(f"모델 로드 실패: {str(e)}")
    
    def free_model(self):
        """모델을 메모리에서 해제합니다."""
        if hasattr(self, 'ctx') and self.ctx:
            try:
                self.dll.whisper_free(self.ctx)
                self.ctx = None
                return True
            except Exception:
                return False
        return True
            
    def check_model_validity(self, model_path):
        """모델 파일의 유효성만 검사합니다. 실제 로드는 하지 않습니다."""
        try:
            # 파일이 존재하는지 확인
            if not os.path.exists(model_path):
                raise Exception(f"모델 파일이 존재하지 않습니다: {model_path}")
            
            # 파일 크기 확인 (최소 크기 체크)
            file_size = os.path.getsize(model_path)
            if file_size < 1024:  # 1KB 미만은 유효한 모델 파일이 아닐 가능성이 높음
                raise Exception(f"모델 파일이 너무 작습니다 ({file_size} bytes). 손상된 파일일 수 있습니다.")
            
            return True
        except Exception as e:
            raise Exception(f"모델 파일 검사 실패: {str(e)}")
    
    def transcribe(self, audio_data, language=None, new_segment_callback=None):
        """오디오 데이터를 텍스트로 변환합니다."""
        if not self.ctx:
            raise Exception("모델이 로드되지 않았습니다")
        
        try:
            # 오디오 데이터를 float32 배열로 변환
            audio_float = audio_data.astype(np.float32)
            
            # 배열을 C 포인터로 변환
            audio_ptr = audio_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # 기본 파라미터 가져오기
            params = WhisperFullParams()
            
            try:
                # 전략 매개변수가 있는 버전 시도
                try:
                    self.dll.whisper_full_default_params_by_ref(ctypes.byref(params), WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY)
                except Exception:
                    # 매개변수 없는 버전 시도
                    self.dll.whisper_full_default_params_by_ref(ctypes.byref(params))
            except Exception:
                # 다른 방법 시도: 포인터 기반 방식
                params_ptr = self.dll.whisper_full_default_params()
                if not params_ptr:
                    raise Exception("whisper_full_default_params가 NULL을 반환했습니다")
                return self._transcribe_with_ptr(audio_float, audio_ptr, params_ptr, language)
            
            # 최적의 매개변수 설정
            params.n_threads = 4                # 스레드 수
            params.audio_ctx = 0                # 오디오 컨텍스트 크기 (0: 전체)
            params.max_len = 0                  # 최대 세그먼트 길이 (0: 제한 없음)
            params.greedy.best_of = 5           # 후보 샘플링 수
            params.beam_search.beam_size = 5    # 빔 크기
            
            # 탐지 임계값 매개변수 조정
            params.no_speech_thold = 0.3        # 무음 임계값 (기본값 0.6에서 낮춤)
            params.entropy_thold = 2.0          # 엔트로피 임계값
            params.logprob_thold = -1.0         # 로그 확률 임계값
            
            # 온도 설정
            params.temperature = 0.0            # 초기 온도 (0: 그리디)
            params.temperature_inc = 0.4        # 온도 증분 (기본값 0.2에서 높임)
            
            # 토큰 타임스탬프 임계값
            params.thold_pt = 0.01              # 토큰 타임스탬프 확률 임계값
            
            # 인쇄 옵션
            params.print_progress = True        # 진행 상황 출력
            params.print_realtime = True        # 실시간 결과 출력 활성화
            params.print_timestamps = True      # 타임스탬프 출력
            
            # 언어 지정이 있는 경우 설정
            if language:
                params.language = language.encode('utf-8')
                params.detect_language = False
            else:
                params.detect_language = True
                
            # 비음성 토큰 억제
            params.suppress_nst = True
            
            # 콜백 설정 (있는 경우)
            if new_segment_callback:
                self.current_callback = new_segment_callback
                params.new_segment_callback = new_segment_callback
                params.new_segment_callback_user_data = None
            
            # 변환 실행
            try:
                result = self.dll.whisper_full(self.ctx, ctypes.byref(params), audio_ptr, len(audio_float))
            except Exception as e:
                raise Exception(f"whisper_full 호출 중 예외 발생: {str(e)}")
            
            if result != 0:
                raise Exception(f"오디오 변환 실패: 코드 {result}")
            
            # 결과 텍스트 가져오기 (세그먼트 조합)
            return self._get_transcription_result()
            
        except Exception as e:
            raise Exception(f"변환 오류: {str(e)}")
    
    def _transcribe_with_ptr(self, audio_float, audio_ptr, params_ptr, language=None):
        """포인터 기반 방식으로 변환 (whisper_full_default_params_by_ref가 실패할 경우)"""
        try:
            # 변환 실행
            result = self.dll.whisper_full(self.ctx, params_ptr, audio_ptr, len(audio_float))
            
            if result != 0:
                raise Exception(f"오디오 변환 실패: 코드 {result}")
            
            # 결과 텍스트 가져오기
            return self._get_transcription_result()
            
        except Exception as e:
            raise Exception(f"변환 오류: {str(e)}")
    
    def _get_transcription_result(self):
        """변환 결과 텍스트를 가져옵니다."""
        text = ""
        n_segments = self.dll.whisper_full_n_segments(self.ctx)
        
        for i in range(n_segments):
            segment_text = self.dll.whisper_full_get_segment_text(self.ctx, i)
            
            # 세그먼트의 시작 및 종료 시간 가져오기
            t0 = self.dll.whisper_full_get_segment_t0(self.ctx, i)
            t1 = self.dll.whisper_full_get_segment_t1(self.ctx, i)
            
            # whisper에서는 시간 값이 10ms 단위로 반환됨
            # 밀리초를 초로 변환 (10ms 단위의 값을 초 단위로)
            t0_sec = t0 // 100
            t1_sec = t1 // 100
            
            # 시, 분, 초 계산
            t0_hour = t0_sec // 3600
            t0_min = (t0_sec % 3600) // 60
            t0_sec = t0_sec % 60
            
            t1_hour = t1_sec // 3600
            t1_min = (t1_sec % 3600) // 60
            t1_sec = t1_sec % 60
            
            # 시간 형식 문자열 생성
            time_str = f"{t0_hour:02d}:{t0_min:02d}:{t0_sec:02d} -> {t1_hour:02d}:{t1_min:02d}:{t1_sec:02d}"
            
            if segment_text:
                decoded_text = segment_text.decode('utf-8', errors='replace')
                text += f"{time_str} {decoded_text}\n"
        
        return text.strip()
    
    def get_current_segments(self):
        """현재까지 인식된 모든 세그먼트를 가져옵니다. (실시간 업데이트용)"""
        if not self.ctx:
            return ""
        
        return self._get_transcription_result()
    
    def __del__(self):
        """객체 소멸 시 리소스 해제"""
        try:
            self.free_model()
        except Exception:
            pass
            
    def initialize(self):
        """DLL 함수 초기화 및 설정"""
        try:
            # Whisper DLL 함수 설정
            # 초기화 및 해제 함수
            self.dll.whisper_init_from_file.argtypes = [ctypes.c_char_p]
            self.dll.whisper_init_from_file.restype = ctypes.c_void_p
            
            self.dll.whisper_free.argtypes = [ctypes.c_void_p]
            self.dll.whisper_free.restype = None
            
            # 파라미터 관련 함수 - 두 가지 버전 모두 정의
            try:
                self.dll.whisper_full_default_params_by_ref.argtypes = [ctypes.POINTER(WhisperFullParams), ctypes.c_int]
                self.dll.whisper_full_default_params_by_ref.restype = None
            except Exception:
                # 매개변수 없는 버전 시도
                self.dll.whisper_full_default_params_by_ref.argtypes = [ctypes.POINTER(WhisperFullParams)]
                self.dll.whisper_full_default_params_by_ref.restype = None
            
            # 포인터 기반 방식도 정의
            self.dll.whisper_full_default_params.argtypes = []
            self.dll.whisper_full_default_params.restype = ctypes.c_void_p
            
            # 전체 처리 파라미터 및 실행 함수
            self.dll.whisper_full.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            self.dll.whisper_full.restype = ctypes.c_int
            
            # 세그먼트 관련 함수
            self.dll.whisper_full_n_segments.argtypes = [ctypes.c_void_p]
            self.dll.whisper_full_n_segments.restype = ctypes.c_int
            
            self.dll.whisper_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.whisper_full_get_segment_text.restype = ctypes.c_char_p
            
            # 타임스탬프 관련 함수 추가
            self.dll.whisper_full_get_segment_t0.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.whisper_full_get_segment_t0.restype = ctypes.c_int64
            
            self.dll.whisper_full_get_segment_t1.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.whisper_full_get_segment_t1.restype = ctypes.c_int64
            
            return True
        except Exception as e:
            raise Exception(f"Whisper DLL 함수 초기화 실패: {str(e)}")