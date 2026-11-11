# 저희가왜마지막이조(TLT) · Speech Noise Reduction Challenge

| 항목 | 내용 |
| --- | --- |
| 팀명 | 저희가왜마지막이조(TLT, TheLastTeam(16조)) |
| 제출 회차 | 1회차 |
| 평가 버전 | Git 태그 `round-01` (커밋 해시는 런어스 제출 글에 함께 기재) |
| 알고리즘 | Wiener Filter (decision-directed 사전 SNR 추정), 딥러닝·머신러닝 미사용 |

## 실행 방법

Python 3.10 이상에서 동작합니다 (Python 3.10, 3.12, 3.13, 3.14에서 확인).

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 테스트 WAV 파일을 input/ 폴더에 넣고 실행
python denoise.py

# 3. 결과는 output/ 폴더에 같은 파일명으로 저장됩니다
```

다른 폴더를 쓰려면 `python denoise.py --input <입력 폴더> --output <출력 폴더>`로 지정합니다.
경로는 모두 저장소 기준 상대 경로이며, 개인 PC의 절대 경로에 의존하지 않습니다.

### 입출력 규칙

- `input/` 아래의 모든 `.wav` 파일(대소문자 무관, 하위 폴더 포함)을 처리합니다.
- 입력 파일마다 **같은 파일명**의 출력 WAV 하나를 `output/`에 만듭니다.
- 출력은 입력과 **같은 샘플링 주파수, 채널 수, 샘플 수, WAV 형식**(예: 16-bit PCM)을 유지합니다.
- 여러 채널이면 채널별로 따로 처리합니다. 16 kHz가 아닌 입력은 창 길이(32 ms)가 같도록 STFT 크기를 맞춥니다.
- 정수 PCM 출력의 피크가 1을 넘으면 클리핑 대신 파일 전체 레벨을 일정하게 낮춥니다.
- 읽을 수 없는 파일이 있어도 중단하지 않고, 경고를 표시한 뒤 입력을 그대로 복사해 출력 파일이 빠지지 않게 합니다.

결과가 규칙을 지키는지 확인하려면 다음을 실행합니다.

```bash
python tools/check_io.py          # input/ 과 output/ 비교
```

## 알고리즘

강의에서 제공한 베이스라인 노트북(`noise_reduction_baselines.ipynb`)의 Wiener Filter를 제출 형식에 맞게 옮긴 것입니다.
16 kHz 입력에서 노트북의 `wiener_filter` 함수와 출력이 샘플 단위로 동일함을 확인했습니다.

1. **STFT:** Hann 창 512 샘플(32 ms), hop 128 샘플(75 % 겹침).
2. **잡음 파워 추정:** 경계 패딩을 제외한 완전한 프레임 중 에너지가 가장 낮은 20 %의 평균 파워를 주파수 방향으로 3-bin 평활화합니다. 파일마다 하나의 값을 사용하며 Noisy 입력만 사용합니다.
3. **이득 계산:** 사후 SNR $\gamma_t = |Y_t|^2 / N$, 사전 SNR $\xi_t = a\,|\hat S_{t-1}|^2/N + (1-a)\max(\gamma_t-1, 0)$ (decision-directed).
   이득 규칙은 `gain_rule`로 고르며, 이득을 곱한 뒤 입력 위상으로 복원합니다.
   - `wiener` (현재 기본값, 1회차 제출본): $G_t = \mathrm{clip}\big(\xi_t/(1+\xi_t),\ g_{\min},\ 1\big)$
   - `lsa` (다음 회차 후보): MMSE-LSA $G_t = \max\big(\tfrac{\xi_t}{1+\xi_t}\exp(\tfrac12 E_1(v_t)),\ g_{\min}\big)$, $v_t = \tfrac{\xi_t}{1+\xi_t}\gamma_t$ (Ephraim & Malah, 1985). 논문처럼 상한은 두지 않습니다.
4. **ISTFT:** overlap-add로 파형을 복원하고 입력 길이에 맞춥니다.

| 파라미터 | 값 | 위치 |
| --- | --- | --- |
| 창 길이 / 겹침 | 32 ms / 75 % | `denoiser/config.py` |
| 잡음 추정 프레임 비율 | 0.20 | `noise_fraction` |
| 이득 규칙 | `wiener` (다음 회차 후보는 `lsa`) | `gain_rule` |
| LSA decision-directed 계수 $a$ | 0.95 | `lsa_smoothing` |
| LSA 최소 이득 $g_{\min}$ | 0.178 (−15 dB) | `lsa_gain_floor` |
| Wiener decision-directed 계수 $a$ | 0.98 | `wiener_smoothing` |
| Wiener 최소 이득 $g_{\min}$ | 0.05 (−26 dB) | `wiener_gain_floor` |
| 잡음 추정 방식 | `fixed` (단계 2 실험은 `spp`, 미채택) | `noise_tracker`, `spp_*` |
| OM-LSA | 끔 (단계 3 실험, 미채택) | `omlsa`, `omlsa_*` |
| 사전 SNR 추정 | `dd` (단계 4 채택안은 `dd_bidir`) | `prior_snr`, `ceps_*`, `ml2d_*` |
| 유성음 위상 복원 | 끔 (단계 5 채택안은 주파수 방향, 600 Hz, 평활·크로스페이드) | `phase_reconstruction`, `phase_*` |
| 협대역 음 모듈 | 끔 (단계 6a 실험, 미채택) | `tonal_module`, `tonal_*` |
| 저역 차단 (바람) | 끔 (단계 6b 채택안은 325 Hz) | `wind_module`, `wind_cutoff_hz` |
| RPCA 잡음 PSD | 끔 (단계 6c 실험, 미채택) | `rpca_module`, `rpca_*` |
| 과도음 잡음 PSD | 끔 (단계 6d 실험, 미채택) | `transient_module`, `transient_*` |

LSA 파라미터는 Train에서 PESQ-WB 기준으로 골랐습니다 (최소 이득 −10/−12/−15/−20/−25 dB × $a$ 0.92/0.95/0.98).
`gain_rule="lsa"`로 바꾸면 1단계 실험(MMSE-LSA)의 출력을 냅니다. 이 스위치를 추가해도 `wiener` 출력은 이전과 바이트 단위로 같습니다.

모든 파라미터는 코드(`denoiser/config.py`)에 포함되어 있으며 별도 설정 파일이 필요 없습니다.

## Validation 평가 결과

1회차(`round-01`) 제출 버전(Wiener)으로 Validation 164개 파일을 평가한 결과입니다 (`results/validation_summary.csv`).
MMSE-LSA(`gain_rule="lsa"`)는 아직 제출하지 않은 실험 결과이며, 아래 "실험 기록"에 따로 정리했습니다.

| 입력 | 파일 수 | 평균 SNR (dB) | 평균 PESQ-WB |
| --- | ---: | ---: | ---: |
| Noisy (처리 전) | 164 | 9.63 | 1.395 |
| 제출 알고리즘 (Wiener) | 164 | 13.12 | 1.599 |
| 개선량 | | +3.49 | +0.204 |

PESQ 계산 실패 파일은 0개입니다. 잡음 종류·목표 SNR·난이도별 결과는 `results/validation_by_*.csv`, 파일별 점수는 `results/validation_per_file.csv`에 있습니다.

평가 방법은 베이스라인 노트북과 같습니다.

- **평가 구간:** `samples.csv`에 `valid_samples` 열이 있으면 노트북처럼 앞에서부터 그 길이까지만 채점하고(잡음 제거는 파일 전체에 적용), 없으면 파일 전체를 채점합니다. 현재 Validation 데이터에는 이 열이 없어 파일 전체를 채점했습니다.
- **SNR:** $10\log_{10}\big(\sum s^2 / \sum (s-\hat s)^2\big)$, 레벨·시간 정렬 없음.
- **PESQ:** 16 kHz wideband (`pesq` 0.0.4, mode `wb`).
- **집계:** 파일별 점수의 단순 평균.

`denoise.py`가 실제로 저장한 WAV 파일을 채점하므로, 16-bit 저장 과정의 영향도 포함됩니다.

### 평가 재현

```bash
pip install -r requirements-eval.txt
python tools/evaluate.py --data-root <noise_reduction_challenge 폴더>
```

`<noise_reduction_challenge 폴더>` 안에 `validation/samples.csv`가 있어야 합니다.
결과는 `results/`에 저장됩니다 (전체 요약, 파일별 점수, 잡음 종류·SNR·난이도별 요약).
Windows에서 `pesq` 설치가 실패하면 `conda install -c conda-forge pesq=0.0.4`를 사용하세요.

## 실험 기록

단계별 실험은 `results/experiments.csv`에 기록하고, 실험별 Train 후보·Validation 결과·짝지은 검정·스펙트로그램은 `results/experiments/<날짜>-<이름>/`에 둡니다.

| 단계 | 변경 | Train PESQ-WB | Validation PESQ-WB | Validation SNR (dB) | 결정 |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | Wiener (1회차 제출본) | 1.604 | 1.599 | 13.12 | 기준 |
| 1 | + 이득 함수 MMSE-LSA (−15 dB, $a$ = 0.95) | 1.763 | 1.730 | 13.57 | 채택, 다음 회차 제출 예정 (Wilcoxon $p$ = 7×10⁻¹³) |
| 2 | + 잡음 추적 MMSE-SPP (양방향, ξ_H1 = 10 dB) | 1.779 | 1.741 | 13.15 | 미채택 (Validation $p$ = 0.56, aircraft_cockpit −0.11) |
| 3 | 1단계 + OM-LSA (Cohen SPP, $G_{\min}$ −15 dB) | 1.729 | 1.690 | 13.13 | 미채택 (Validation −0.040) |
| 4a | 1단계 + 켑스트럼-시간 평활 사전 SNR (논문 기본형) | 1.726 | | | 미채택 (Train −0.037) |
| 4b | 1단계 + 양방향 DD 사전 SNR (오프라인 변형) | 1.771 | 1.739 | 13.83 | 채택, 청취 확인 대기 (Wilcoxon $p$ = 7×10⁻⁵) |
| 5 | 4b + 유성음 위상 복원 (주파수 방향, 600 Hz 이하, 피치 평활·크로스페이드) | 1.774 | 1.743 | 13.78 | 채택 (효과 작음, 청취 확인 대기; Wilcoxon $p$ = 0.004) |
| 6a | 4b + 협대역 음 추적 잡음 PSD (경보·사이렌 대상) | 1.763 | 1.726 | 12.95 | 미채택 (Validation −0.013) |
| 6b | 5 + 저역 차단 (325 Hz 아래 이득을 하한으로, 바람·저역 잡음 대상) | 1.872 | 1.805 | 3.83 | 채택 (Wilcoxon $p$ = 6×10⁻¹¹), 청취 확인 대기, SNR 크게 하락 |
| 6c | 6b + RPCA 저랭크 잡음 PSD (기계·엔진·조종석 대상) | 1.870 | 1.805 | 3.83 | 미채택 (Validation −0.000) |
| 6d | 6b + HPSS 과도음 잡음 PSD (박수·클릭 대상) | 1.873 | 1.809 | 3.82 | 미채택 (유의하지 않음, 대상 파일 1–3개) |

### 로드맵 결과 요약

로드맵 단계 1–6을 모두 실험했습니다. 채택된 최종 설정은 다음과 같고, Validation 평균 PESQ-WB는 1회차 1.599에서 **1.805** (+0.206)입니다.

```python
Config(gain_rule="lsa", prior_snr="dd_bidir",                       # 단계 1, 4b
       phase_reconstruction=True, phase_mode="frequency",           # 단계 5
       phase_smooth=True, phase_fade_frames=3, phase_max_hz=600.0,
       wind_module=True, wind_cutoff_hz=325.0)                      # 단계 6b
```

- `config.py` 기본값은 아직 1회차 제출본(Wiener)입니다. 다음 회차 제출 때 위 값으로 바꿉니다.
- 단계 4b, 5, 6b는 청취 확인 전입니다. 특히 6b는 325 Hz 아래(남성 목소리의 기본 주파수 대역)를 줄이므로 목소리가 얇게 들릴 수 있고, SNR이 3.8 dB로 처리 전(9.6 dB)보다 낮아집니다. 순위 지표인 PESQ-WB는 모든 주요 잡음 종류에서 올랐습니다.
- SNR을 함께 지켜야 한다면 6b 없이(PESQ 1.743, SNR 13.8 dB) 또는 낮은 차단 주파수(150 Hz: Train PESQ 1.789, SNR 8.7 dB)를 쓸 수 있습니다.

단계 2는 1단계 설정 위에 잡음 PSD를 프레임마다 추적하는 unbiased MMSE-SPP(Gerkmann & Hendriks, 2012)를 더한 실험입니다 (`noise_tracker="spp"`, 기본값은 `"fixed"`).
논문 그대로의 인과 추적(VOICEBOX 기본값)은 Train에서 1단계보다 0.064 낮았고, 비인과(양방향) 변형만 Train에서 +0.016 올랐습니다.
Validation에서는 유의하지 않았고 정상 잡음(조종석·광대역)이 나빠져 채택하지 않았습니다. 바람 잡음은 두 데이터 모두에서 좋아졌습니다 (Validation +0.068).

단계 3은 1단계 설정(고정 잡음 추정) 위에서 이득을 OM-LSA $G = G_{H1}^{\,p}\,G_{\min}^{\,1-p}$(Cohen & Berdugo, 2001)로 바꾼 실험입니다 (`omlsa=True`, 기본값 `False`).
음성 존재 확률 $p$를 Cohen 방식과 단계 2의 SPP 두 가지로, $G_{\min}$을 −10~−25 dB로 바꿔 Train 18개 후보를 비교했지만 모두 1단계보다 낮았고 (최고 −0.033), Validation에서도 −0.040 ($p$ = 7×10⁻¹⁶)이라 채택하지 않았습니다.

단계 4는 1단계 설정에서 사전 SNR 추정만 바꾼 실험입니다 (`prior_snr`, 기본값 `"dd"`).
논문(Breithaupt 외, 2008)의 켑스트럼-시간 평활(`"cepstral"`)은 Table 1 상수 그대로와 변형 7개 모두 Train에서 1단계보다 낮았습니다 (최고 −0.008).
오프라인 변형 중 정방향·역방향 decision-directed 추정을 평균하는 `"dd_bidir"`만 Train +0.008, Validation +0.009 ($p$ = 7×10⁻⁵, 파일 67 % 개선)로 두 데이터에서 같은 방향으로 좋아져 채택했습니다.
문헌 근거가 없는 비인과 처리이며, 개선 폭이 작아 청취 확인 후 다음 회차 제출 설정(`gain_rule="lsa"`, `prior_snr="dd_bidir"`)에 넣습니다.

단계 5는 4b 위에서 유성음 프레임의 위상을 기본 주파수로 복원하는 실험입니다 (Krawczyk & Gerkmann, 2014; `phase_reconstruction`, 기본값 `False`).
논문 방식(시간·주파수 결합, 4 kHz까지)은 실제 피치 궤적에서 시간 방향 전파가 어긋나 Train −0.016이었습니다.
주파수 방향만 복원하고, 피치 궤적 평활과 유성음 경계 크로스페이드(오프라인 변형)를 더하고, 복원 대역을 600 Hz 이하로 줄였을 때만 Train +0.003, Validation +0.004 ($p$ = 0.004)로 좋아졌습니다.
채택 조건은 모두 통과했지만 개선 폭이 매우 작아, 청취 확인 결과에 따라 다음 회차 제출 설정에 넣을지 정합니다.

단계 6a는 개선 폭이 가장 작은 잡음 종류인 경보·사이렌(`alarm_siren`, Validation에서 처리 전 대비 +0.27로 파일 10개 이상 종류 중 최저)을 대상으로, 클립 전체에서 오래 이어지는 협대역 피크 궤적을 찾아 잡음 PSD에 더하는 모듈입니다 (`tonal_module`, 기본값 `False`).
Train 후보 11개가 모두 4b보다 낮았고 (최고 −0.008), Validation에서도 −0.013 (SNR −0.9 dB)이라 채택하지 않았습니다.
정상적인 경보음은 이미 고정 잡음 추정에 들어 있어 사이렌에서의 이득이 작았고 (+0.007, 유의하지 않음), 오래 이어지는 저주파 음성 하모닉이 잡음으로 잘못 잡혀 다른 잡음 종류에서 음성이 손상되었습니다.

단계 6b–6d는 남은 잡음 종류별 모듈을 단계 5 설정 위에 하나씩 더한 실험입니다.
- **6b 저역 차단 (`wind_module`):** 차단 주파수 아래 대역의 이득을 LSA 하한(−15 dB)으로 제한합니다. Train에서 80–400 Hz를 비교해 325 Hz가 가장 좋았고 (+0.098), Validation +0.062로 파일 10개 이상인 모든 잡음 종류가 올라 채택했습니다. 음성의 기본 주파수 대역도 줄어들어 SNR은 3.8 dB로 크게 떨어집니다.
- **6c RPCA (`rpca_module`):** 크기 스펙트로그램의 저랭크 성분을 잡음 PSD로 쓰는 방식입니다. 논문의 기본 λ에서는 음성까지 저랭크 성분에 들어가 잡음을 약 3배 과대추정했고, λ를 줄인 후보도 모두 6b보다 낮아 채택하지 않았습니다.
- **6d 과도음 (`transient_module`):** HPSS 타악기 성분을 잡음 PSD에 더합니다. 대상인 박수·클릭·문 소리는 크게 좋아졌지만 (Validation clap +0.22, click +0.15) 파일이 1–3개뿐이고 전체 효과가 유의하지 않아 채택하지 않았습니다.

```bash
# 설정을 바꿔 평가 (결과를 실험 폴더에 저장)
python tools/evaluate.py --data-root <데이터 폴더> --split train --set gain_rule=lsa \
    --set lsa_gain_floor=0.178 --results-dir results/experiments/<이름>/train-<후보>
# 두 평가 결과의 파일별 짝지은 비교 (Wilcoxon signed-rank, 잡음 종류·SNR별)
python tools/compare_runs.py <기준 결과 폴더> <새 결과 폴더> --split validation
# 고정 청취 목록(results/listening_set.txt)의 스펙트로그램 (matplotlib 필요)
python tools/plot_spectrograms.py --data-root <데이터 폴더> --run "이전=<폴더>" --run "새=<폴더>" --out <폴더>
```

## 폴더 구조

```
denoise.py              실행 파일 (input/ → output/)
denoiser/
  config.py             모든 파라미터
  stft.py               STFT / ISTFT
  noise_estimation.py   잡음 파워 추정 (파일 고정값 / MMSE-SPP 추적)
  gains.py              Wiener / MMSE-LSA / OM-LSA 이득 (decision-directed)
  phase.py              유성음 위상 복원 (YIN 피치, 고조파 모델)
  pipeline.py           신호·파일·폴더 단위 처리
tools/
  check_io.py           입출력 규칙 검사
  evaluate.py           Train / Validation SNR / PESQ 평가 (--set 으로 설정 덮어쓰기)
  compare_runs.py       두 평가 결과의 짝지은 검정, 잡음 종류·SNR별 비교
  plot_spectrograms.py  고정 청취 목록의 스펙트로그램
requirements.txt        실행에 필요한 패키지
requirements-eval.txt   평가에만 필요한 패키지
input/  output/         테스트 입력 / 결과 폴더
```

## 참고 문헌

- Y. Ephraim and D. Malah, "Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 32, no. 6, pp. 1109–1121, 1984. (decision-directed 사전 SNR 추정)
- Y. Ephraim and D. Malah, "Speech Enhancement Using a Minimum Mean-Square Error Log-Spectral Amplitude Estimator," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 33, no. 2, pp. 443–445, 1985, DOI 10.1109/TASSP.1985.1164550. (MMSE-LSA 이득)
- T. Gerkmann and R. C. Hendriks, "Unbiased MMSE-Based Noise Power Estimation With Low Complexity and Low Tracking Delay," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 20, no. 4, pp. 1383–1393, 2012, DOI 10.1109/TASL.2011.2180896. (단계 2 실험, 미채택)
- I. Cohen and B. Berdugo, "Speech enhancement for non-stationary noise environments," *Signal Processing*, vol. 81, no. 11, pp. 2403–2418, 2001, DOI 10.1016/S0165-1684(01)00128-1. (단계 3 실험, 미채택)
- C. Breithaupt, T. Gerkmann, and R. Martin, "A novel a priori SNR estimation approach based on selective cepstro-temporal smoothing," *Proc. IEEE ICASSP*, pp. 4897–4900, 2008, DOI 10.1109/ICASSP.2008.4518755. (단계 4, 저자 공개 PDF로 수식과 Table 1 상수 대조)
- M. Krawczyk and T. Gerkmann, "STFT Phase Reconstruction in Voiced Speech for an Improved Single-Channel Speech Enhancement," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 22, no. 12, pp. 1931–1940, 2014, DOI 10.1109/TASLP.2014.2354236. (단계 5, 저자 공개 PDF로 수식 대조)
- A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for speech and music," *Journal of the Acoustical Society of America*, vol. 111, no. 4, pp. 1917–1930, 2002. (단계 5의 피치 추정, 학습 없음)
- E. J. Candès, X. Li, Y. Ma, and J. Wright, "Robust principal component analysis?," *Journal of the ACM*, vol. 58, no. 3, pp. 1–37, 2011, DOI 10.1145/1970392.1970395. (단계 6c, 미채택; dganguli/robust-pca (MIT)는 대조용으로 읽기만 함)
- P.-S. Huang, S. D. Chen, P. Smaragdis, and M. Hasegawa-Johnson, "Singing-voice separation from monaural recordings using robust principal component analysis," *Proc. IEEE ICASSP*, pp. 57–60, 2012, DOI 10.1109/ICASSP.2012.6287816. (단계 6c, 크기 스펙트로그램에 RPCA 적용)
- D. FitzGerald, "Harmonic/percussive separation using median filtering," *Proc. DAFx*, 2010. (단계 6d, 미채택; DOI 없음, 상수 미대조)
- R. Talmon, I. Cohen, and S. Gannot, "Transient noise reduction using nonlocal diffusion filters," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 19, no. 6, pp. 1584–1599, 2011, DOI 10.1109/TASL.2010.2093651. (단계 6d 배경)
- R. J. McAulay and T. F. Quatieri, "Speech analysis/synthesis based on a sinusoidal representation," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 34, no. 4, pp. 744–754, 1986, DOI 10.1109/TASSP.1986.1164910. (단계 6 피크 궤적 연결 방식, 미채택; 상수는 자체 설정)
- 단계 3의 음성 부재 확률 상수와 OM-LSA decision-directed 방식: chenwj1989/python-speech-enhancement `pns/suppression_gain.py` (커밋 f38fde7, BSD-3, Copyright (c) 2019 wjchen)를 읽고 직접 구현.
- 이득 수식 대조용으로 읽은 코드 (복사하지 않고 직접 구현): wilsonchingg/logmmse `logmmse/logmmse.py` (MIT), ImperialCollegeLondon/sap-voicebox `voicebox/v_ssubmmse.m`, `voicebox/v_estnoiseg.m` (LGPL-3, 단계 2 상수의 출처).
- 코드 출처: 강의 제공 베이스라인 노트북 `noise_reduction_baselines.ipynb`의 `stft_audio`, `inverse_stft`, `estimate_noise_power`, `wiener_filter`.
