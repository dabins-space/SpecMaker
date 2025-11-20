# webspec_maker_streamlit.py
# -*- coding: utf-8 -*-
"""
PDF → '제품명, 썸네일, 제품 설명(가변), 제품 요약(가변), 특징(불릿)' 생성
- GPT: 전부/부분(제품명/설명/요약/특징) 개별 생성 버튼
- 설명/요약 최대 글자수 UI에서 설정
- 특징: '- ' 불릿, 리스트 직전에 "Features" 라벨 출력
- 특징 GPT 생성: 한국어로 자연스럽게, 숫자/단위/규격 최대 포함
  '있으면 반드시 포함 / 없으면 생략(추측 금지)', 자리채움("정보 없음") 미삽입
- 규칙 기반도 유지(키 없이도 동작, 확장된 카테고리 동의어 기반 '있으면 포함')
- API Key 상태 표시 + 설정/변경
- 이미지: 추출/리스트 클릭 미리보기/썸네일 저장(200x200, 300x300 포함)
- 되돌리기: 다단계 Undo + '처음 상태로' 복원 + 임의 시점 스냅샷
- 수작업 편집 후 저장(Markdown)

실행:
    pip install streamlit pymupdf pillow openai
    streamlit run app.py
"""

import os
import re
import io
import json
import unicodedata
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
from PIL import Image

APP_TITLE = "WebSpec Maker v7.1"
VERSION = "v7.1.0"

# ===== 기본 설정(초기값) =====
DESC_MAX_DEFAULT = 40
SUMMARY_MAX_DEFAULT = 200
FEATURE_MAX = 24  # 특징 최대 줄 수

# (확장) 필수 카테고리 (있으면 반드시 포함 대상)
REQUIRED_KEYS = [
    "CPU", "GPU", "Memory", "Storage", "Power", "I/O", "LAN",
    "Dimensions", "Operating Temperature"
]

# 카테고리 동의어(키/값 어디에 나와도 매칭)
KEY_ALIASES = {
    "CPU": ["CPU", "Processor", "프로세서"],
    "GPU": ["GPU", "Graphics", "그래픽", "VGA"],
    "Memory": ["Memory", "RAM", "메모리", "DRAM", "DDR", "RDIMM", "UDIMM", "ECC"],
    "Storage": ["Storage", "스토리지", "SSD", "HDD", "NVMe", "SATA", "M.2", "U.2"],
    "Power": ["Power", "PSU", "전원", "AC", "DC", "Adapter", "어댑터", "전력", "입력"],
    "I/O": ["I/O", "IO", "Interface", "입출력", "포트", "USB", "PCIe", "HDMI", "DP", "VGA", "COM", "RS-232", "RS-485"],
    "LAN": ["LAN", "Ethernet", "GbE", "10GbE", "2.5GbE", "RJ-45", "네트워크"],
    "Dimensions": ["Dimensions", "크기", "규격", "외형", "치수", "Size", "Form Factor", "W x D x H", "WxDxH", "mm", "cm", "inch"],
    "Operating Temperature": ["Operating Temperature", "Operating Temp", "동작 온도", "작동 온도", "온도", "Temperature range", "Operating range"]
}

TARGET_SIZES = ["200x200", "300x300", "600x600", "800x800", "1200x900", "1200x630", "1920x1080"]
BULLET_MARKERS = ("•", "●", "-", "▪", "‣", "–", "—", "·", "*")
MODEL_RX = re.compile(r"\b([A-Z]{2,}[A-Z0-9\-_/]{1,}|[A-Z0-9]{2,}\-[A-Z0-9\-_/]+)\b", re.IGNORECASE)

# ================== GPT ==================
GPT_DEFAULT_MODEL = "gpt-4o-mini"

def init_session_state():
    """세션 상태 초기화"""
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = None
    if 'openai_ready' not in st.session_state:
        st.session_state.openai_ready = False
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = None
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = ""
    if 'images' not in st.session_state:
        st.session_state.images = []
    if 'selected_image_idx' not in st.session_state:
        st.session_state.selected_image_idx = None
    if 'selected_image_indices' not in st.session_state:
        st.session_state.selected_image_indices = []
    if 'last_thumb_path' not in st.session_state:
        st.session_state.last_thumb_path = None
    if 'undo_stack' not in st.session_state:
        st.session_state.undo_stack = []
    if 'initial_state' not in st.session_state:
        st.session_state.initial_state = None
    if 'var_name' not in st.session_state:
        st.session_state.var_name = ""
    if 'var_desc' not in st.session_state:
        st.session_state.var_desc = ""
    if 'var_summary' not in st.session_state:
        st.session_state.var_summary = ""
    if 'var_feats' not in st.session_state:
        st.session_state.var_feats = ""
    if 'desc_max' not in st.session_state:
        st.session_state.desc_max = DESC_MAX_DEFAULT
    if 'summary_max' not in st.session_state:
        st.session_state.summary_max = SUMMARY_MAX_DEFAULT
    if 'temp_dir' not in st.session_state:
        # Streamlit Cloud 호환: 현재 디렉토리 사용
        try:
            # 임시 디렉토리 생성 시도
            st.session_state.temp_dir = tempfile.mkdtemp()
        except Exception:
            # 실패 시 현재 디렉토리의 temp 폴더 사용
            st.session_state.temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(st.session_state.temp_dir, exist_ok=True)

def try_load_openai(api_key: str = None):
    """OpenAI 클라이언트 초기화"""
    try:
        from openai import OpenAI
        # 순서: 직접 입력 > Streamlit secrets > 직접 파일 읽기 > 환경변수
        key = None
        
        if api_key:
            key = api_key.strip()
        else:
            # Streamlit Cloud/Secrets에서 먼저 확인 (가장 우선순위)
            try:
                if hasattr(st, 'secrets') and st.secrets is not None:
                    # Streamlit Cloud: st.secrets는 딕셔너리처럼 직접 접근 가능
                    try:
                        # 방법 1: 직접 딕셔너리 접근 (가장 확실)
                        if 'OPENAI_API_KEY' in st.secrets:
                            key = str(st.secrets['OPENAI_API_KEY']).strip()
                    except (TypeError, AttributeError, KeyError):
                        try:
                            # 방법 2: get 메서드 사용
                            key = str(st.secrets.get('OPENAI_API_KEY', '')).strip()
                        except (TypeError, AttributeError):
                            try:
                                # 방법 3: to_dict() 변환 후 접근
                                secrets_dict = st.secrets.to_dict()
                                if secrets_dict and 'OPENAI_API_KEY' in secrets_dict:
                                    key = str(secrets_dict['OPENAI_API_KEY']).strip()
                            except:
                                pass
            except Exception:
                # Streamlit Cloud가 아닌 환경이면 무시하고 계속 진행
                pass
            
            # 로컬 파일에서 확인 (.streamlit/secrets.toml)
            if not key:
                try:
                    secrets_path = os.path.join('.streamlit', 'secrets.toml')
                    if os.path.exists(secrets_path):
                        with open(secrets_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # TOML 파싱: OPENAI_API_KEY = "값"
                            match = re.search(r'OPENAI_API_KEY\s*=\s*["\']([^"\']+)["\']', content)
                            if match:
                                key = match.group(1).strip()
                except Exception:
                    pass
            
            # 환경변수에서 확인
            if not key:
                key = os.environ.get("OPENAI_API_KEY", "").strip()
        
        # API Key 검증
        if not key or len(key.strip()) == 0:
            st.session_state.openai_ready = False
            st.session_state.openai_client = None
            return False
        
        key = key.strip()
        
        # API Key 형식 검증 (sk-로 시작하는지 확인)
        if not key.startswith('sk-'):
            st.session_state.openai_ready = False
            st.session_state.openai_client = None
            return False
        
        # OpenAI 클라이언트 초기화
        st.session_state.openai_client = OpenAI(api_key=key)
        st.session_state.openai_ready = True
        return True
    except Exception as e:
        st.session_state.openai_ready = False
        st.session_state.openai_client = None
        return False

def ensure_openai_ready():
    """OpenAI API가 준비되었는지 확인"""
    if st.session_state.openai_ready:
        return True
    st.error("OpenAI API Key가 설정되지 않았습니다. .streamlit/secrets.toml 파일이나 환경변수 OPENAI_API_KEY를 설정하세요.")
    return False

def _truncate(s: str, n: int) -> str:
    return (s or "").strip().replace("\n", " ")[:max(0, n)]

def _extract_json_str(mixed_text: str) -> str:
    s = (mixed_text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if m: return m.group(1).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start: return s[start:end+1].strip()
    return s

def _gpt(prompt: str, model: str = GPT_DEFAULT_MODEL, temperature: float = 0.25, max_out: int = 2000) -> str:
    """GPT API 호출"""
    if not st.session_state.openai_ready:
        raise RuntimeError("OpenAI API가 준비되지 않았습니다.")
    
    client = st.session_state.openai_client
    # Responses API → fallback Chat Completions
    try:
        resp = client.responses.create(
            model=model, input=prompt,
            temperature=temperature, max_output_tokens=max_out
        )
        return (getattr(resp, "output_text", "") or "").strip()
    except Exception:
        pass
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature, max_tokens=max_out
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise RuntimeError(f"GPT 호출 실패: {e}")

# ---- 한국어 불릿 정규화 유틸 ----
_HANGUL_RX = re.compile(r"[\uac00-\ud7a3]")
_UNIT_RX = re.compile(r"(?i)\b(ghz|mhz|khz|gb|mb|tb|w|v|a|mm|cm|inch|gbit|gbe|pcie|usb|sata|nvme|ddr|rdimm|udimm|ecc|wifi|bt|poe)\b")
_PUNCT_TRIM_RX = re.compile(r"[\"'·•*•]+$")

def _normalize_korean_bullets(items, max_len=64):
    """
    - 모든 항목은 '- '로 시작
    - 이모지/장식문자/겹공백 제거
    - 끝의 불필요한 구두점 제거
    - 길이 컷
    - 중복 제거(대소문자/공백 무시)
    - 한국어 토큰 또는 숫자/단위가 없으면 제외(영문 설명성 문구 제거)
    """
    out = []
    seen = set()
    for raw in items:
        s = (raw or "").strip()
        if not s:
            continue
        # 접두 보정
        if s.startswith(("-", "•", "*")):
            s = s.lstrip("•* ").strip()
        if not s.startswith("- "):
            s = "- " + s

        # 공백/장식 정리
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"\s+", " ", s)
        s = s.replace("•", "").replace("–", "-").replace("—", "-")
        s = _PUNCT_TRIM_RX.sub("", s).strip()
        s = re.sub(r"[、,…]+$", "", s).strip()

        # 길이 컷
        if len(s) > max_len:
            s = s[:max_len].rstrip()

        # 한국어/숫자·단위 체크
        has_kr = bool(_HANGUL_RX.search(s))
        has_num_unit = bool(re.search(r"\d", s) or _UNIT_RX.search(s))
        if not has_kr and not has_num_unit:
            continue

        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

# ================== GPT: 필드별 생성 ==================
def gpt_generate_name(seed_text: str) -> str:
    prompt = (
        "아래 데이터시트에서 실제 제품 모델명/시리즈명을 한 줄로만 추출하세요.\n"
        "문장/설명 없이 모델명만 반환:\n\n"
        f"{seed_text[:1500]}"
    )
    return _truncate(_gpt(prompt, max_out=80).splitlines()[0], 100)

def gpt_generate_description(seed_text: str, max_chars: int) -> str:
    prompt = (
        f"다음 데이터시트를 읽고, 제품을 가장 잘 설명하는 한국어 한 줄 설명을 {max_chars}자 이내 명사구로 작성하세요.\n"
        "조건: 마침표/따옴표/이모지 금지, 핵심 용도/형태/규격 포함\n\n"
        f"{seed_text[:2000]}\n\n정답:"
    )
    return _truncate(_gpt(prompt, max_out=120).splitlines()[0].strip(" \"'"), max_chars)

def gpt_generate_summary(seed_text: str, max_chars: int) -> str:
    prompt = (
        f"다음 데이터시트의 핵심을 한국어로 {max_chars}자 이내 한 문장으로 요약하세요. "
        "가능하면 CPU/GPU/메모리/스토리지/전원/네트워크/I/O 등 구체 수치나 규격을 포함하세요.\n\n"
        f"{seed_text[:2500]}\n\n정답:"
    )
    return _truncate(_gpt(prompt, max_out=240), max_chars)

def gpt_generate_features(seed_text: str, max_items: int):
    """
    문서 내용을 최대한 포함한 한국어 불릿 리스트 생성.
    - 각 항목은 '- ' 접두
    - 다음 카테고리 정보가 문서에 있으면 반드시 포함(없으면 생략, 추측 금지):
      CPU, GPU, Memory, Storage, Power, I/O, LAN, Dimensions, Operating Temperature
    - 가능한 한 실제 수치/단위/포트개수/전력/규격 포함
    - 한국어로만(모델명/규격/숫자·단위는 원문 허용)
    - 각 불릿 64자 이내 명사구, 과장어 금지
    """
    req = ", ".join(REQUIRED_KEYS)
    prompt = (
        "당신은 한국어 기술 마케터입니다. 아래 데이터시트를 바탕으로 제품 '특징'을 한국어 불릿 리스트로 작성하세요.\n"
        f"- 항목 수 최대 {max_items}개, 가능한 한 많은 핵심 정보를 담되 중복 금지\n"
        "- 각 항목은 '- '로 시작(그 외 기호 금지)\n"
        "- 각 항목은 64자 이내, 간결한 명사구. 문장부호·이모지·마케팅 수사(혁신적/탁월 등) 금지\n"
        "- 실측/규격/포트개수/전력/크기 등 숫자·단위를 최대한 포함 (예: 2×10GbE, DDR5 512GB, 2U, 800W 등)\n"
        f"- 다음 카테고리 정보가 문서에 **있으면 반드시** 최소 1항목 포함: {req}. 문서에 **명시가 없으면 생략**(추측/창작 금지)\n"
        "출력은 불릿들만 줄바꿈으로 구분하여 제공하세요(설명/코드블록 금지).\n\n"
        f"[원문]\n{seed_text[:3500]}\n\n정답(불릿만):"
    )
    raw = _gpt(prompt, max_out=1400)
    feats = _normalize_korean_bullets(raw.splitlines(), max_len=64)
    return feats[:max_items]

# ================== 규칙 기반(비-GPT) ==================
def read_pdf_text(pdf_bytes: bytes) -> str:
    """PDF에서 텍스트 추출"""
    texts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            texts.append(page.get_text("text"))
    return "\n".join(texts)

def guess_title_from_pdf(pdf_bytes: bytes, text: str) -> str:
    """PDF에서 제품명 추정"""
    # 메타데이터 → 큰 폰트 → 패턴 → 첫 줄
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            meta = doc.metadata or {}
            for k in ("title","subject"):
                v = (meta.get(k) or "").strip()
                if v and 2 <= len(v) <= 100:
                    m = MODEL_RX.search(v)
                    return (m.group(1) if m else v).strip()
    except Exception:
        pass
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            tops = []
            for i, page in enumerate(doc):
                if i > 1: break
                blocks = page.get_text("dict").get("blocks", [])
                for b in blocks:
                    for l in b.get("lines", []):
                        for s in l.get("spans", []):
                            txt = (s.get("text") or "").strip()
                            size = float(s.get("size") or 0)
                            if txt and size >= 8:
                                tops.append((size, txt))
            if tops:
                tops.sort(key=lambda x: x[0], reverse=True)
                for _, t in tops[:50]:
                    m = MODEL_RX.search(t)
                    if m: return m.group(1).strip()
                return tops[0][1][:80]
    except Exception:
        pass
    m2 = MODEL_RX.search(text)
    if m2: return m2.group(1)[:80]
    for line in text.splitlines():
        s = line.strip()
        if s: return s[:80]
    return "제품명"

def extract_kv_candidates(text: str, max_items: int = 500) -> list:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or len(s) > 300: continue
        if ":" in s: k, v = s.split(":", 1)
        elif " - " in s: k, v = s.split(" - ", 1)
        else: continue
        k, v = k.strip(), v.strip()
        if 1 <= len(k) <= 60 and v:
            out.append((k, v))
        if len(out) >= max_items: break
    return out

def extract_bullets(text: str, max_items: int = 300) -> list:
    feats=[]
    for line in text.splitlines():
        s=line.strip()
        if not s: continue
        if s[0] in BULLET_MARKERS or any(s.startswith(m+" ") for m in BULLET_MARKERS):
            s2 = s.lstrip("".join(BULLET_MARKERS)).strip(" -–—·:*")
            if s2: feats.append(s2)
        if len(feats) >= max_items: break
    return feats

def _matches_key(alias_key: str, k: str, v: str) -> bool:
    """k나 v 어딘가에 해당 카테고리 동의어가 포함되면 매칭으로 간주"""
    kk = (k or "").lower()
    vv = (v or "").lower()
    for tok in KEY_ALIASES.get(alias_key, [alias_key]):
        t = tok.lower()
        if t in kk or t in vv:
            return True
    return False

def rules_build_fields(pdf_bytes: bytes, text: str, desc_max: int, summary_max: int, limit: int = FEATURE_MAX):
    """
    규칙 기반:
    - name: 제목/큰 폰트/패턴 추정
    - desc/summary: 앞부분에서 지정 길이 컷
    - 특징: 확장된 카테고리(있으면 반드시 포함), 없으면 생략.
            이후 KV/불릿로 보강 → 한국어 불릿 정규화
    """
    name = guess_title_from_pdf(pdf_bytes, text)
    desc = _truncate(text, desc_max)
    summary = _truncate(text, summary_max)

    kv = extract_kv_candidates(text, max_items=500)
    bullets = extract_bullets(text, max_items=300)

    feats = []
    used = set()

    # (확장) 카테고리 '있으면 포함'
    for key in REQUIRED_KEYS:
        for i, (k, v) in enumerate(kv):
            if i in used: continue
            if _matches_key(key, k, v):
                feats.append(f"- {key}: {v}")
                used.add(i)
                break
        if len(feats) >= limit:
            return name, desc, summary, _normalize_korean_bullets(feats, 64)[:limit]

    # 추가 K:V
    for i, (k, v) in enumerate(kv):
        if i in used: continue
        pair = f"- {k}: {v}"
        if pair not in feats:
            feats.append(pair)
        if len(feats) >= limit:
            return name, desc, summary, _normalize_korean_bullets(feats, 64)[:limit]

    # 불릿 보강
    for b in bullets:
        bl = f"- {b}" if not b.startswith("-") else b
        if bl not in feats:
            feats.append(bl)
        if len(feats) >= limit: break

    feats = _normalize_korean_bullets(feats, 64)
    return name, desc, summary, feats[:limit]

# ================== 이미지 유틸 ==================
def ensure_output_dir(base_dir: str, pdf_name: str) -> str:
    name = os.path.splitext(pdf_name)[0]
    out = os.path.join(base_dir, f"{name}_assets")
    os.makedirs(out, exist_ok=True)
    return out

def extract_images(pdf_bytes_or_path, out_dir: str) -> list:
    """PDF에서 모든 이미지 추출 (중복 허용, 모든 이미지 저장)"""
    saved = []
    
    # 출력 디렉토리 확인 및 생성
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            st.error(f"출력 디렉토리 생성 실패: {e}")
            return saved
    
    # PDF 파일을 열기 (bytes 또는 경로 모두 지원)
    doc = None
    try:
        if isinstance(pdf_bytes_or_path, bytes):
            if len(pdf_bytes_or_path) == 0:
                st.error("PDF 파일 데이터가 비어있습니다.")
                return saved
            doc = fitz.open(stream=pdf_bytes_or_path, filetype="pdf")
        else:
            if not os.path.exists(pdf_bytes_or_path):
                st.error(f"PDF 파일을 찾을 수 없습니다: {pdf_bytes_or_path}")
                return saved
            doc = fitz.open(pdf_bytes_or_path)
    except Exception as e:
        st.error(f"PDF 파일 열기 실패: {e}")
        return saved
    
    if doc is None or doc.page_count == 0:
        st.warning("PDF 파일에 페이지가 없습니다.")
        return saved
    
    try:
        # 페이지별로 이미지 추출 (중복 체크하지 않고 모든 이미지 저장)
        img_counter = 1
        total_images_found = 0
        
        for pno, page in enumerate(doc, start=1):
            try:
                # get_images(full=True)로 모든 이미지 정보 가져오기
                images = page.get_images(full=True)
                total_images_found += len(images)
                
                if len(images) == 0:
                    continue
                
                for idx, img_info in enumerate(images, start=1):
                    xref = img_info[0]
                    try:
                        base = doc.extract_image(xref)
                        img_bytes = base.get("image")
                        
                        if not img_bytes:
                            continue
                        
                        img_ext = base.get("ext", "png")  # 이미지 확장자 (png, jpg 등)
                        
                        # 이미지 로드 및 저장
                        im = Image.open(io.BytesIO(img_bytes))
                        # RGBA로 변환하여 투명도 지원
                        if im.mode != 'RGBA':
                            im = im.convert("RGBA")
                        
                        # 파일명: 페이지번호_이미지인덱스_전체순번.png
                        save_name = f"p{pno:02d}_img{idx:02d}_#{img_counter:03d}.png"
                        save_path = os.path.join(out_dir, save_name)
                        im.save(save_path, format="PNG")
                        saved.append(save_path)
                        img_counter += 1
                    except Exception as e:
                        # 개별 이미지 추출 실패는 경고만 표시
                        pass
            except Exception as e:
                st.warning(f"페이지 {pno} 처리 중 오류: {e}")
        
        # 결과 요약
        if len(saved) == 0:
            if total_images_found == 0:
                st.warning(f"⚠️ PDF에서 이미지를 찾을 수 없습니다. (총 {doc.page_count}페이지 검색)")
            else:
                st.warning(f"⚠️ 이미지를 찾았지만 추출하지 못했습니다. (발견: {total_images_found}개, 추출: 0개)")
        else:
            st.info(f"📊 이미지 발견: {total_images_found}개, 성공적으로 추출: {len(saved)}개")
            
    except Exception as e:
        st.error(f"이미지 추출 중 오류 발생: {e}")
    finally:
        if doc:
            doc.close()
    
    return saved

def pad_resize(image: Image.Image, target_w: int, target_h: int, bg_rgb=(255,255,255)) -> Image.Image:
    im = image.convert("RGBA")
    im.thumbnail((target_w, target_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (target_w, target_h), bg_rgb + (255,))
    x = (target_w - im.width) // 2
    y = (target_h - im.height) // 2
    canvas.paste(im, (x, y), im)
    return canvas.convert("RGB")

# ================== 내보내기(Markdown) ==================
def export_markdown(name: str, thumb_path: str, desc: str, summary: str, feats: list) -> str:
    lines = []
    lines.append(f"**제품명**: {name}")
    lines.append(f"**썸네일**: {os.path.basename(thumb_path) if thumb_path else ''}")
    lines.append("")
    lines.append("### 제품 설명")
    lines.append(desc)
    lines.append("")
    lines.append("### 제품 요약")
    lines.append(summary)
    lines.append("")
    lines.append("### 특징")
    lines.append("**◆ 주요 특징 ◆**")  # 블릿 직전 라벨(굵게 표시)
    if feats:
        lines.extend(feats)
    else:
        lines.append("-")
    return "\n".join(lines)

# ================== 상태 관리 ==================
def get_state():
    """현재 상태 가져오기"""
    return (
        st.session_state.var_name,
        st.session_state.var_desc,
        st.session_state.var_summary,
        st.session_state.var_feats,
        st.session_state.last_thumb_path
    )

def set_state(state):
    """상태 설정"""
    n, d, s, f, t = state
    st.session_state.var_name = n
    st.session_state.var_desc = d
    st.session_state.var_summary = s
    st.session_state.var_feats = f
    st.session_state.last_thumb_path = t

def push_undo():
    """Undo 스택에 현재 상태 저장"""
    st.session_state.undo_stack.append(get_state())

def undo_once():
    """한 단계 되돌리기"""
    if not st.session_state.undo_stack:
        st.warning("되돌릴 기록이 없습니다")
        return False
    state = st.session_state.undo_stack.pop()
    set_state(state)
    st.success(f"되돌리기 완료 (남은 단계: {len(st.session_state.undo_stack)})")
    return True

def undo_to_initial():
    """처음 상태로 복원"""
    if st.session_state.initial_state is None:
        st.warning("초기 상태가 없습니다")
        return False
    set_state(st.session_state.initial_state)
    st.session_state.undo_stack.clear()
    st.success("처음 상태로 복원 완료")
    return True

# ================== 페이지 설정 (반드시 최상단에) ==================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 메인 앱 ==================
def main():
    init_session_state()
    
    # API Key 자동 로드 (Streamlit secrets > 환경변수, 화면에 표시하지 않음)
    if not st.session_state.openai_ready:
        loaded = try_load_openai()
        # 디버깅: Streamlit Cloud에서 API Key 로딩 실패 시 안내 (한 번만)
        if not loaded and 'api_key_warning_shown' not in st.session_state:
            st.session_state.api_key_warning_shown = True
            # Streamlit Cloud 환경인지 확인
            try:
                if hasattr(st, 'secrets') and st.secrets is not None:
                    # Secrets가 있지만 API Key가 없는 경우
                    if 'OPENAI_API_KEY' not in st.secrets:
                        st.warning("⚠️ Streamlit Cloud Secrets에 OPENAI_API_KEY가 설정되지 않았습니다. Settings → Secrets에서 API Key를 추가해주세요.")
            except:
                pass
    
    # 사이드바
    with st.sidebar:
        st.title("⚙️ 빠른 액션")
        
        # API 상태 표시
        if st.session_state.openai_ready:
            st.success("✅ API 활성")
        else:
            st.error("❌ API 비활성")
        
        st.divider()
        
        # PDF 상태 표시
        if st.session_state.pdf_name:
            st.info(f"📄 {st.session_state.pdf_name}")
        else:
            st.info("📄 PDF를 업로드하세요")
        
        st.divider()
        
        # 빠른 설정
        with st.expander("⚙️ 설정", expanded=False):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.session_state.desc_max = st.number_input(
                    "설명 최대자",
                    min_value=10,
                    max_value=200,
                    value=st.session_state.desc_max,
                    step=5
                )
            with col_s2:
                st.session_state.summary_max = st.number_input(
                    "요약 최대자",
                    min_value=50,
                    max_value=600,
                    value=st.session_state.summary_max,
                    step=10
                )
        
        st.divider()
        
        # Undo 기능
        st.subheader("↩️ 되돌리기")
        if st.button("↶ 한 단계 되돌리기", use_container_width=True):
            undo_once()
        
        if st.button("⏮️ 처음 상태로", use_container_width=True):
            undo_to_initial()
        
        if st.button("📸 현재 상태 스냅샷", use_container_width=True):
            push_undo()
            st.success("스냅샷 저장 완료")
        
        st.divider()
        
        # 저장
        st.subheader("💾 저장")
        name, desc, summ, feats, thumb = get_state()
        feats_list = [l.strip() if l.strip().startswith("-") else f"- {l.strip()}"
                     for l in feats.splitlines() if l.strip()]
        feats_list = _normalize_korean_bullets(feats_list, 64)[:FEATURE_MAX]
        md_content = export_markdown(name, thumb, desc, summ, feats_list)
        
        st.download_button(
            label="📥 Markdown 저장",
            data=md_content,
            file_name=f"{st.session_state.var_name or '제품정보'}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    # 메인 영역
    st.title(f"{APP_TITLE} {VERSION}")
    
    # PDF 업로드 섹션 (메인 상단)
    col_upload_main1, col_upload_main2 = st.columns([3, 1])
    
    with col_upload_main1:
        uploaded_file = st.file_uploader("📄 PDF 파일 선택", type=["pdf"], help="PDF 파일을 업로드하세요")
    
    with col_upload_main2:
        if st.session_state.pdf_path and st.button("🖼️ 이미지 추출", use_container_width=True, type="secondary"):
            with st.spinner("이미지 추출 중..."):
                saved = extract_images(
                    st.session_state.pdf_path,
                    st.session_state.output_dir
                )
                st.session_state.images = saved
                st.success(f"이미지 {len(saved)}개 추출 완료")
                st.rerun()
    
    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            # 새 PDF 로드
            pdf_bytes = uploaded_file.read()
            st.session_state.pdf_path = pdf_bytes
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.output_dir = ensure_output_dir(
                st.session_state.temp_dir, 
                uploaded_file.name
            )
            
            try:
                # 텍스트 추출
                with st.spinner("PDF 처리 중..."):
                    st.session_state.raw_text = read_pdf_text(pdf_bytes)
                    
                    # 규칙 기반 필드 생성
                    name, desc, summary, feats = rules_build_fields(
                        pdf_bytes,
                        st.session_state.raw_text,
                        st.session_state.desc_max,
                        st.session_state.summary_max,
                        FEATURE_MAX
                    )
                    
                    # 상태 설정
                    st.session_state.var_name = name
                    st.session_state.var_desc = desc
                    st.session_state.var_summary = summary
                    st.session_state.var_feats = "\n".join(feats)
                    
                    # 초기 상태 저장
                    st.session_state.initial_state = get_state()
                    st.session_state.undo_stack.clear()
                
                st.success(f"✅ PDF 로드 완료: {uploaded_file.name}")
                st.rerun()
                
            except Exception as e:
                st.error(f"PDF 읽기 실패: {e}")
    
    # GPT 생성 버튼들 (메인 상단)
    if st.session_state.raw_text:
        st.divider()
        st.subheader("🤖 GPT 자동 생성")
        
        if not st.session_state.openai_ready:
            st.warning("⚠️ OpenAI API Key가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인하세요.")
        else:
            col_gpt1, col_gpt2, col_gpt3, col_gpt4, col_gpt5 = st.columns(5)
            
            with col_gpt1:
                if st.button("📝 제품명", use_container_width=True, type="secondary"):
                    if ensure_openai_ready():
                        push_undo()
                        try:
                            with st.spinner("제품명 생성 중..."):
                                name = gpt_generate_name(st.session_state.raw_text)
                                st.session_state.var_name = name
                            st.success("✅ 제품명 생성 완료")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 제품명 생성 실패: {e}")
            
            with col_gpt2:
                if st.button("📄 설명", use_container_width=True, type="secondary"):
                    if ensure_openai_ready():
                        push_undo()
                        try:
                            with st.spinner("설명 생성 중..."):
                                desc = gpt_generate_description(
                                    st.session_state.raw_text,
                                    st.session_state.desc_max
                                )
                                st.session_state.var_desc = desc
                            st.success("✅ 설명 생성 완료")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 설명 생성 실패: {e}")
            
            with col_gpt3:
                if st.button("📋 요약", use_container_width=True, type="secondary"):
                    if ensure_openai_ready():
                        push_undo()
                        try:
                            with st.spinner("요약 생성 중..."):
                                summary = gpt_generate_summary(
                                    st.session_state.raw_text,
                                    st.session_state.summary_max
                                )
                                st.session_state.var_summary = summary
                            st.success("✅ 요약 생성 완료")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 요약 생성 실패: {e}")
            
            with col_gpt4:
                if st.button("✨ 특징", use_container_width=True, type="secondary"):
                    if ensure_openai_ready():
                        push_undo()
                        try:
                            with st.spinner("특징 생성 중... (시간이 걸릴 수 있습니다)"):
                                feats = gpt_generate_features(
                                    st.session_state.raw_text,
                                    FEATURE_MAX
                                )
                                st.session_state.var_feats = "\n".join(feats)
                            st.success("✅ 특징 생성 완료")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 특징 생성 실패: {e}")
            
            with col_gpt5:
                if st.button("🚀 전부 생성", use_container_width=True, type="primary"):
                    if ensure_openai_ready():
                        push_undo()
                        try:
                            with st.spinner("모든 필드 생성 중... (시간이 걸릴 수 있습니다)"):
                                st.session_state.var_name = gpt_generate_name(st.session_state.raw_text)
                                st.session_state.var_desc = gpt_generate_description(
                                    st.session_state.raw_text,
                                    st.session_state.desc_max
                                )
                                st.session_state.var_summary = gpt_generate_summary(
                                    st.session_state.raw_text,
                                    st.session_state.summary_max
                                )
                                feats = gpt_generate_features(st.session_state.raw_text, FEATURE_MAX)
                                st.session_state.var_feats = "\n".join(feats)
                            st.success("✅ 전부 생성 완료!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ GPT 전부 생성 실패: {e}")
    
    st.divider()
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📝 제품 정보 편집", "🖼️ 이미지 관리", "📋 미리보기"])
    
    with tab1:
        # 필드 편집
        st.header("📝 제품 정보 편집")
        
        if not st.session_state.pdf_path:
            st.info("💡 PDF를 업로드하면 자동으로 기본 정보가 생성됩니다.")
        
        # 제품명
        col_name_label, col_name_copy = st.columns([11, 1], gap="small")
        with col_name_label:
            st.markdown("### 제품명")
        with col_name_copy:
            if st.session_state.var_name:
                name_text_js = json.dumps(st.session_state.var_name)
                # 예쁜 복사 버튼 생성 (data 속성 사용, 줄 바꿈 유지)
                # 줄 바꿈을 특수 마커로 임시 치환했다가 JavaScript에서 복원
                name_text_escaped = st.session_state.var_name.replace('\\', '\\\\').replace('"', '&quot;').replace("'", "&#39;").replace('\n', '[[NEWLINE]]')
                copy_btn_html = f"""
                <div style="padding: 0.25rem 0;">
                    <button data-text="{name_text_escaped}" 
                            onclick="(function(evt) {{
                                const btn = evt.target || evt.currentTarget || this;
                                let text = btn.getAttribute('data-text') || '';
                                text = text.replace(/\[\[NEWLINE\]\]/g, '\\n');
                                const origHtml = btn.innerHTML;
                                const origStyle = btn.style.cssText;
                                
                                function showSuccess() {{
                                    btn.innerHTML = '✓ 복사됨';
                                    btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                                    btn.style.boxShadow = '0 4px 15px rgba(16, 185, 129, 0.4)';
                                    btn.style.transform = 'scale(1.05)';
                                    setTimeout(function() {{
                                        btn.innerHTML = origHtml;
                                        btn.style.cssText = origStyle;
                                    }}, 1500);
                                }}
                                
                                function fallbackCopy() {{
                                    const ta = document.createElement('textarea');
                                    ta.value = text;
                                    ta.style.position = 'fixed';
                                    ta.style.top = '0';
                                    ta.style.left = '0';
                                    ta.style.width = '2em';
                                    ta.style.height = '2em';
                                    ta.style.padding = '0';
                                    ta.style.border = 'none';
                                    ta.style.opacity = '0';
                                    document.body.appendChild(ta);
                                    ta.focus();
                                    ta.select();
                                    try {{
                                        if (document.execCommand('copy')) {{
                                            showSuccess();
                                        }} else {{
                                            alert('복사 실패: 브라우저가 클립보드 접근을 허용하지 않습니다.');
                                        }}
                                    }} catch(e) {{
                                        alert('복사 실패');
                                    }}
                                    document.body.removeChild(ta);
                                }}
                                
                                if (navigator.clipboard && navigator.clipboard.writeText) {{
                                    navigator.clipboard.writeText(text).then(function() {{
                                        showSuccess();
                                    }}).catch(function() {{
                                        fallbackCopy();
                                    }});
                                }} else {{
                                    fallbackCopy();
                                }}
                            }})(event || window.event)" 
                            onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(99, 102, 241, 0.35)';"
                            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(99, 102, 241, 0.25)';"
                    style="width: 100%; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           color: white; 
                           border: none; 
                           border-radius: 8px; 
                           padding: 6px 10px; 
                           cursor: pointer; 
                           font-size: 11px;
                           font-weight: 600;
                           box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
                           transition: all 0.3s ease;
                           letter-spacing: 0.3px;" 
                    title="클릭하면 복사됩니다">
                        📋 복사
                    </button>
                </div>
                """
                components.html(copy_btn_html, height=35)
        
        st.session_state.var_name = st.text_input(
            "",
            value=st.session_state.var_name,
            placeholder="제품명을 입력하세요",
            label_visibility="collapsed",
            key="name_input_field"
        )
        
        # 제품 설명
        col_desc_label, col_desc_copy = st.columns([11, 1], gap="small")
        with col_desc_label:
            st.markdown("### 제품 설명 (최대 {}자)".format(st.session_state.desc_max))
        with col_desc_copy:
            if st.session_state.var_desc:
                desc_text_js = json.dumps(st.session_state.var_desc)
                # 예쁜 복사 버튼 생성 (data 속성 사용, 줄 바꿈 유지)
                # 줄 바꿈을 특수 마커로 임시 치환했다가 JavaScript에서 복원
                desc_text_escaped = st.session_state.var_desc.replace('\\', '\\\\').replace('"', '&quot;').replace("'", "&#39;").replace('\n', '[[NEWLINE]]')
                copy_btn_html = f"""
                <div style="padding: 0.25rem 0;">
                    <button data-text="{desc_text_escaped}" 
                            onclick="(function(evt) {{
                                const btn = evt.target || evt.currentTarget || this;
                                let text = btn.getAttribute('data-text') || '';
                                text = text.replace(/\[\[NEWLINE\]\]/g, '\\n');
                                const origHtml = btn.innerHTML;
                                const origStyle = btn.style.cssText;
                                
                                function showSuccess() {{
                                    btn.innerHTML = '✓ 복사됨';
                                    btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                                    btn.style.boxShadow = '0 4px 15px rgba(16, 185, 129, 0.4)';
                                    btn.style.transform = 'scale(1.05)';
                                    setTimeout(function() {{
                                        btn.innerHTML = origHtml;
                                        btn.style.cssText = origStyle;
                                    }}, 1500);
                                }}
                                
                                function fallbackCopy() {{
                                    const ta = document.createElement('textarea');
                                    ta.value = text;
                                    ta.style.position = 'fixed';
                                    ta.style.top = '0';
                                    ta.style.left = '0';
                                    ta.style.width = '2em';
                                    ta.style.height = '2em';
                                    ta.style.padding = '0';
                                    ta.style.border = 'none';
                                    ta.style.opacity = '0';
                                    document.body.appendChild(ta);
                                    ta.focus();
                                    ta.select();
                                    try {{
                                        if (document.execCommand('copy')) {{
                                            showSuccess();
                                        }} else {{
                                            alert('복사 실패: 브라우저가 클립보드 접근을 허용하지 않습니다.');
                                        }}
                                    }} catch(e) {{
                                        alert('복사 실패');
                                    }}
                                    document.body.removeChild(ta);
                                }}
                                
                                if (navigator.clipboard && navigator.clipboard.writeText) {{
                                    navigator.clipboard.writeText(text).then(function() {{
                                        showSuccess();
                                    }}).catch(function() {{
                                        fallbackCopy();
                                    }});
                                }} else {{
                                    fallbackCopy();
                                }}
                            }})(event || window.event)" 
                            onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(99, 102, 241, 0.35)';"
                            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(99, 102, 241, 0.25)';"
                    style="width: 100%; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           color: white; 
                           border: none; 
                           border-radius: 8px; 
                           padding: 6px 10px; 
                           cursor: pointer; 
                           font-size: 11px;
                           font-weight: 600;
                           box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
                           transition: all 0.3s ease;
                           letter-spacing: 0.3px;" 
                    title="클릭하면 복사됩니다">
                        📋 복사
                    </button>
                </div>
                """
                components.html(copy_btn_html, height=35)
        
        st.session_state.var_desc = st.text_area(
            "",
            value=st.session_state.var_desc,
            height=80,
            help=f"현재: {len(st.session_state.var_desc)}자 / 최대: {st.session_state.desc_max}자",
            key="desc_area",
            label_visibility="collapsed"
        )
        
        # 제품 요약
        col_sum_label, col_sum_copy = st.columns([11, 1], gap="small")
        with col_sum_label:
            st.markdown("### 제품 요약 (최대 {}자)".format(st.session_state.summary_max))
        with col_sum_copy:
            if st.session_state.var_summary:
                summary_text_js = json.dumps(st.session_state.var_summary)
                # 예쁜 복사 버튼 생성 (data 속성 사용, 줄 바꿈 유지)
                # 줄 바꿈을 특수 마커로 임시 치환했다가 JavaScript에서 복원
                summary_text_escaped = st.session_state.var_summary.replace('\\', '\\\\').replace('"', '&quot;').replace("'", "&#39;").replace('\n', '[[NEWLINE]]')
                copy_btn_html = f"""
                <div style="padding: 0.25rem 0;">
                    <button data-text="{summary_text_escaped}" 
                            onclick="(function(evt) {{
                                const btn = evt.target || evt.currentTarget || this;
                                let text = btn.getAttribute('data-text') || '';
                                text = text.replace(/\[\[NEWLINE\]\]/g, '\\n');
                                const origHtml = btn.innerHTML;
                                const origStyle = btn.style.cssText;
                                
                                function showSuccess() {{
                                    btn.innerHTML = '✓ 복사됨';
                                    btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                                    btn.style.boxShadow = '0 4px 15px rgba(16, 185, 129, 0.4)';
                                    btn.style.transform = 'scale(1.05)';
                                    setTimeout(function() {{
                                        btn.innerHTML = origHtml;
                                        btn.style.cssText = origStyle;
                                    }}, 1500);
                                }}
                                
                                function fallbackCopy() {{
                                    const ta = document.createElement('textarea');
                                    ta.value = text;
                                    ta.style.position = 'fixed';
                                    ta.style.top = '0';
                                    ta.style.left = '0';
                                    ta.style.width = '2em';
                                    ta.style.height = '2em';
                                    ta.style.padding = '0';
                                    ta.style.border = 'none';
                                    ta.style.opacity = '0';
                                    document.body.appendChild(ta);
                                    ta.focus();
                                    ta.select();
                                    try {{
                                        if (document.execCommand('copy')) {{
                                            showSuccess();
                                        }} else {{
                                            alert('복사 실패: 브라우저가 클립보드 접근을 허용하지 않습니다.');
                                        }}
                                    }} catch(e) {{
                                        alert('복사 실패');
                                    }}
                                    document.body.removeChild(ta);
                                }}
                                
                                if (navigator.clipboard && navigator.clipboard.writeText) {{
                                    navigator.clipboard.writeText(text).then(function() {{
                                        showSuccess();
                                    }}).catch(function() {{
                                        fallbackCopy();
                                    }});
                                }} else {{
                                    fallbackCopy();
                                }}
                            }})(event || window.event)" 
                            onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(99, 102, 241, 0.35)';"
                            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(99, 102, 241, 0.25)';"
                    style="width: 100%; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           color: white; 
                           border: none; 
                           border-radius: 8px; 
                           padding: 6px 10px; 
                           cursor: pointer; 
                           font-size: 11px;
                           font-weight: 600;
                           box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
                           transition: all 0.3s ease;
                           letter-spacing: 0.3px;" 
                    title="클릭하면 복사됩니다">
                        📋 복사
                    </button>
                </div>
                """
                components.html(copy_btn_html, height=35)
        
        st.session_state.var_summary = st.text_area(
            "",
            value=st.session_state.var_summary,
            height=120,
            help=f"현재: {len(st.session_state.var_summary)}자 / 최대: {st.session_state.summary_max}자",
            key="summary_area",
            label_visibility="collapsed"
        )
        
        # 특징
        col_feat_label, col_feat_copy = st.columns([11, 1], gap="small")
        with col_feat_label:
            st.markdown("### 특징 (최대 {}줄, '- ' 불릿)".format(FEATURE_MAX))
        with col_feat_copy:
            if st.session_state.var_feats:
                # 복사할 때 "◆ 주요 특징 ◆" 포함
                feats_with_label = f"**◆ 주요 특징 ◆**\n{st.session_state.var_feats}"
                feats_text_js = json.dumps(feats_with_label)
                # 예쁜 복사 버튼 생성 (data 속성 사용, 줄 바꿈 유지)
                # 줄 바꿈을 특수 마커로 임시 치환했다가 JavaScript에서 복원
                feats_text_escaped = feats_with_label.replace('\\', '\\\\').replace('"', '&quot;').replace("'", "&#39;").replace('\n', '[[NEWLINE]]')
                copy_btn_html = f"""
                <div style="padding: 0.25rem 0;">
                    <button data-text="{feats_text_escaped}" 
                            onclick="(function(evt) {{
                                const btn = evt.target || evt.currentTarget || this;
                                let text = btn.getAttribute('data-text') || '';
                                text = text.replace(/\[\[NEWLINE\]\]/g, '\\n');
                                const origHtml = btn.innerHTML;
                                const origStyle = btn.style.cssText;
                                
                                function showSuccess() {{
                                    btn.innerHTML = '✓ 복사됨';
                                    btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                                    btn.style.boxShadow = '0 4px 15px rgba(16, 185, 129, 0.4)';
                                    btn.style.transform = 'scale(1.05)';
                                    setTimeout(function() {{
                                        btn.innerHTML = origHtml;
                                        btn.style.cssText = origStyle;
                                    }}, 1500);
                                }}
                                
                                function fallbackCopy() {{
                                    const ta = document.createElement('textarea');
                                    ta.value = text;
                                    ta.style.position = 'fixed';
                                    ta.style.top = '0';
                                    ta.style.left = '0';
                                    ta.style.width = '2em';
                                    ta.style.height = '2em';
                                    ta.style.padding = '0';
                                    ta.style.border = 'none';
                                    ta.style.opacity = '0';
                                    document.body.appendChild(ta);
                                    ta.focus();
                                    ta.select();
                                    try {{
                                        if (document.execCommand('copy')) {{
                                            showSuccess();
                                        }} else {{
                                            alert('복사 실패: 브라우저가 클립보드 접근을 허용하지 않습니다.');
                                        }}
                                    }} catch(e) {{
                                        alert('복사 실패');
                                    }}
                                    document.body.removeChild(ta);
                                }}
                                
                                if (navigator.clipboard && navigator.clipboard.writeText) {{
                                    navigator.clipboard.writeText(text).then(function() {{
                                        showSuccess();
                                    }}).catch(function() {{
                                        fallbackCopy();
                                    }});
                                }} else {{
                                    fallbackCopy();
                                }}
                            }})(event || window.event)" 
                            onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(99, 102, 241, 0.35)';"
                            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(99, 102, 241, 0.25)';"
                    style="width: 100%; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           color: white; 
                           border: none; 
                           border-radius: 8px; 
                           padding: 6px 10px; 
                           cursor: pointer; 
                           font-size: 11px;
                           font-weight: 600;
                           box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
                           transition: all 0.3s ease;
                           letter-spacing: 0.3px;" 
                    title="클릭하면 복사됩니다">
                        📋 복사
                    </button>
                </div>
                """
                components.html(copy_btn_html, height=35)
        
        st.session_state.var_feats = st.text_area(
            "",
            value=st.session_state.var_feats,
            height=400,
            help="각 항목은 '- '로 시작하세요",
            key="feats_area",
            label_visibility="collapsed"
        )
        
    
    with tab2:
        st.header("🖼️ 이미지 관리")
        
        if not st.session_state.pdf_path:
            st.warning("⚠️ 먼저 PDF를 업로드해주세요 (1번 탭)")
            st.stop()
        
        if not st.session_state.images:
            st.info("💡 PDF를 업로드한 후 '이미지 추출' 버튼을 클릭하면 이미지를 추출할 수 있습니다.")
            st.stop()
        
        # 선택된 이미지 초기화
        if 'selected_image_indices' not in st.session_state:
            st.session_state.selected_image_indices = []
        
        # 이미지 그리드 표시 (4개씩 한 줄)
        st.subheader(f"📸 추출된 이미지 ({len(st.session_state.images)}개)")
        
        # 선택된 이미지 관리
        selected_indices = st.session_state.selected_image_indices.copy()
        
        # 이미지 그리드 (4개씩)
        for i in range(0, len(st.session_state.images), 4):
            cols = st.columns(4)
            for j in range(4):
                idx = i + j
                if idx < len(st.session_state.images):
                    img_path = st.session_state.images[idx]
                    img_name = os.path.basename(img_path)
                    
                    with cols[j]:
                        try:
                            img = Image.open(img_path)
                            # 썸네일 생성 (작게 표시)
                            thumb = img.copy()
                            thumb.thumbnail((150, 150), Image.LANCZOS)
                            
                            # 체크박스
                            is_selected = idx in selected_indices
                            checked = st.checkbox(
                                f"선택",
                                value=is_selected,
                                key=f"img_check_{idx}",
                                label_visibility="collapsed"
                            )
                            
                            # 이미지 표시
                            st.image(thumb, caption=img_name, use_container_width=True)
                            st.caption(f"{img.width}×{img.height}")
                            
                            # 체크박스 상태 업데이트
                            if checked and idx not in selected_indices:
                                selected_indices.append(idx)
                            elif not checked and idx in selected_indices:
                                selected_indices.remove(idx)
                        except Exception as e:
                            st.error(f"로드 오류: {e}")
                            st.text(img_name)
        
        # 선택 상태 업데이트
        st.session_state.selected_image_indices = sorted(selected_indices)
        
        # 선택된 이미지 작업 섹션
        if st.session_state.selected_image_indices:
            st.divider()
            st.subheader(f"✅ 선택된 이미지 ({len(st.session_state.selected_image_indices)}개)")
            
            # 썸네일 설정
            col_size, col_bg = st.columns(2)
            with col_size:
                target_size = st.selectbox("목표 크기", TARGET_SIZES, index=0, key="thumb_size")
            with col_bg:
                bg_color = st.color_picker("배경색", "#FFFFFF", key="thumb_bg")
            
            # 썸네일 생성 및 다운로드 버튼
            col_gen, col_dl = st.columns(2)
            
            with col_gen:
                if st.button("🎨 썸네일 생성", use_container_width=True, type="primary"):
                    try:
                        # 크기 파싱
                        m = re.match(r"(\d+)\s*x\s*(\d+)", target_size)
                        if not m:
                            tw, th = 600, 600
                        else:
                            tw, th = int(m.group(1)), int(m.group(2))
                        
                        # 배경색 변환
                        bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
                        
                        # 첫 번째 선택된 이미지만 썸네일 생성 (1개만)
                        if st.session_state.selected_image_indices:
                            idx = st.session_state.selected_image_indices[0]
                            img_path = st.session_state.images[idx]
                            try:
                                with Image.open(img_path) as im:
                                    out = pad_resize(im, tw, th, bg_rgb)
                                
                                base = os.path.splitext(os.path.basename(img_path))[0]
                                out_path = os.path.join(st.session_state.output_dir, f"{base}_thumb_{tw}x{th}.jpg")
                                out.save(out_path, quality=92)
                                st.session_state.last_thumb_path = out_path
                                
                                st.success(f"✅ 썸네일 생성 완료!")
                                # 썸네일 미리보기 (작게 표시)
                                preview = out.copy()
                                preview.thumbnail((300, 300), Image.LANCZOS)
                                st.image(preview, caption=f"생성된 썸네일 ({tw}x{th})", width=300)
                                st.rerun()
                            except Exception as e:
                                st.error(f"썸네일 생성 실패: {e}")
                        else:
                            st.warning("이미지를 선택해주세요.")
                    except Exception as e:
                        st.error(f"썸네일 생성 실패: {e}")
            
            with col_dl:
                # 썸네일 다운로드 버튼 (1개만)
                if st.session_state.last_thumb_path and os.path.exists(st.session_state.last_thumb_path):
                    with open(st.session_state.last_thumb_path, "rb") as f:
                        st.download_button(
                            label="📥 썸네일 다운로드",
                            data=f.read(),
                            file_name=os.path.basename(st.session_state.last_thumb_path),
                            mime="image/jpeg",
                            use_container_width=True,
                            key="thumb_download"
                        )
                else:
                    st.info("💡 썸네일을 먼저 생성해주세요")
        
        # 전체 이미지 다운로드
        st.divider()
        st.subheader("📦 전체 이미지 다운로드")
        
        if st.button("📥 모든 이미지 ZIP 다운로드", use_container_width=True):
            try:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for img_path in st.session_state.images:
                        if os.path.exists(img_path):
                            zip_file.write(img_path, os.path.basename(img_path))
                    
                    # 생성된 썸네일 찾아서 추가
                    if st.session_state.output_dir and os.path.exists(st.session_state.output_dir):
                        for file in os.listdir(st.session_state.output_dir):
                            if 'thumb' in file.lower() and file.endswith('.jpg'):
                                thumb_path = os.path.join(st.session_state.output_dir, file)
                                if os.path.exists(thumb_path):
                                    zip_file.write(thumb_path, file)
                
                zip_buffer.seek(0)
                zip_data = zip_buffer.read()
                zip_name = f"{os.path.splitext(st.session_state.pdf_name or 'images')[0]}_all_images.zip"
                
                st.session_state.zip_data = zip_data
                st.session_state.zip_name = zip_name
                st.success(f"ZIP 파일 생성 완료: {len(zip_data)} bytes")
                st.rerun()
            except Exception as e:
                st.error(f"ZIP 생성 실패: {e}")
        
        # 전체 ZIP 다운로드 버튼
        if 'zip_data' in st.session_state and 'zip_name' in st.session_state:
            st.download_button(
                label="📥 전체 이미지 ZIP 다운로드",
                data=st.session_state.zip_data,
                file_name=st.session_state.zip_name,
                mime="application/zip",
                use_container_width=True,
                key="all_zip_download"
            )
    
    with tab3:
        st.header("📋 미리보기 & 저장")
        
        name, desc, summ, feats, thumb = get_state()
        feats_list = [l.strip() if l.strip().startswith("-") else f"- {l.strip()}"
                     for l in feats.splitlines() if l.strip()]
        feats_list = _normalize_korean_bullets(feats_list, 64)[:FEATURE_MAX]
        md_content = export_markdown(name, thumb, desc, summ, feats_list)
        
        # 미리보기
        st.subheader("📄 Markdown 미리보기")
        st.text_area(
            "미리보기 (복사 가능)",
            md_content,
            height=500,
            key="preview_area",
            help="이 영역의 텍스트를 복사해서 사용할 수 있습니다"
        )
        
        # 저장 버튼
        st.divider()
        st.subheader("💾 파일 저장")
        
        col_save1, col_save2 = st.columns(2)
        
        with col_save1:
            st.download_button(
                label="📥 Markdown 저장",
                data=md_content,
                file_name=f"{st.session_state.var_name or '제품정보'}.md",
                mime="text/markdown",
                use_container_width=True,
                type="primary"
            )
        
        with col_save2:
            # 복사 버튼 (클립보드)
            if st.button("📋 클립보드에 복사", use_container_width=True, type="secondary"):
                st.code(md_content, language="markdown")
                st.success("위의 코드를 선택해서 복사하세요")
        
        # 정보 요약
        st.divider()
        st.subheader("📊 생성 정보 요약")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("제품명", st.session_state.var_name[:30] + "..." if len(st.session_state.var_name) > 30 else st.session_state.var_name or "없음")
        
        with col_info2:
            st.metric("설명 글자수", f"{len(desc)} / {st.session_state.desc_max}")
        
        with col_info3:
            st.metric("요약 글자수", f"{len(summ)} / {st.session_state.summary_max}")
        
        st.metric("특징 항목 수", len(feats_list))

if __name__ == "__main__":
    main()

