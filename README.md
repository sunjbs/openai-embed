## A-Z 가이드: Poetry로 Python 프로젝트 시작하기

### Zed Setting

``` settings.json
{
  "languages": {
    "Python": {
      "language_servers": ["ruff"],
      "format_on_save": "on",
      "formatter": [
        {
          "code_actions": {
            "source.organizeImports.ruff": true,
            "source.fixAll.ruff": true
          }
        },
        {
          "language_server": {
            "name": "ruff"
          }
        }
      ]
    }
  }
}
```

### 0단계: Poetry 설치 (아직 설치하지 않았다면)

먼저 Poetry가 시스템에 설치되어 있어야 합니다. 터미널(PowerShell, cmd, bash 등)을 열고 아래 명령어를 실행하세요.

**macOS / Linux:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

설치가 완료되면 터미널을 재시작한 후, 아래 명령어로 설치를 확인합니다.
```bash
poetry --version
```
버전 정보가 출력되면 성공적으로 설치된 것입니다.

---

### 1단계: 새 프로젝트 생성 (`poetry new`)

가장 기본적인 방법은 `poetry new` 명령어를 사용하는 것입니다. 이 명령어는 표준적인 프로젝트 구조를 자동으로 만들어줍니다.

```bash
poetry new my-awesome-project
poetry config virtualenvs.in-project true
```

위 명령어를 실행하면 `my-awesome-project`라는 이름의 폴더가 생성됩니다.
그리고, 가상환경을 프로젝트 폴더 내에 생성하도록 설정합니다.

---

### 2단계: 생성된 프로젝트 구조 이해하기

`my-awesome-project` 폴더로 이동(`cd my-awesome-project`)해서 내부 구조를 살펴보면 다음과 같습니다.

```
my-awesome-project/
├── pyproject.toml         # 👈 가장 중요한 파일! 프로젝트 설정 및 의존성 관리
├── README.md              # 프로젝트 설명 파일
├── my_awesome_project/    # 실제 파이썬 패키지 소스 코드가 위치하는 폴더
│   └── __init__.py
└── tests/                 # 테스트 코드가 위치하는 폴더
    └── __init__.py
```

*   **`pyproject.toml`**: Poetry의 핵심 파일입니다. 프로젝트의 이름, 버전, 설명 같은 메타데이터와 `requests`, `pandas` 같은 라이브러리 의존성 목록이 여기에 기록됩니다.
*   **`my_awesome_project/`**: 프로젝트 이름(하이픈 `-`은 언더스코어 `_`로 변경됨)과 동일한 이름의 패키지 폴더입니다. 여러분의 파이썬 코드는 주로 이 폴더 안에 작성하게 됩니다.
*   **`tests/`**: `pytest`와 같은 도구를 사용하여 작성한 테스트 코드를 넣는 곳입니다.

---

### 3단계: 의존성(라이브러리) 추가하기 (`poetry add`)

프로젝트에 필요한 라이브러리를 추가해 보겠습니다. 예를 들어, 웹 요청을 보내는 데 사용되는 `requests` 라이브러리를 추가해 봅시다.

프로젝트 폴더 안에서 다음 명령어를 실행합니다.

```bash
poetry add requests
```

이 명령어를 실행하면 Poetry는 다음 작업들을 **자동으로** 수행합니다.

1.  **가상 환경 생성**: 이 프로젝트만을 위한 격리된 Python 환경을 만듭니다.
2.  **라이브러리 설치**: `requests`와 그에 필요한 다른 라이브러리들을 가상 환경에 설치합니다.
3.  **`pyproject.toml` 업데이트**: `[tool.poetry.dependencies]` 섹션에 `requests`를 추가합니다.
4.  **`poetry.lock` 파일 생성/업데이트**: 설치된 모든 라이브러리의 정확한 버전 정보를 기록하여, 다른 환경에서도 동일한 버전으로 설치할 수 있도록 보장합니다.

개발 중에만 필요한 라이브러리(예: 테스트 도구 `pytest`)는 `--group dev` 옵션을 사용해 추가합니다.

```bash
poetry add pytest --group dev
```

---

### 4단계: 코드 실행하기 (`poetry run` & `poetry shell`)

Poetry가 관리하는 가상 환경 안에서 코드를 실행해야 합니다. 두 가지 방법이 있습니다.

#### 방법 1: `poetry run` 사용하기

가장 간단한 방법입니다. 실행하려는 명령어 앞에 `poetry run`을 붙이면 됩니다.

먼저, `my_awesome_project/main.py` 파일을 하나 만들고 아래 코드를 작성해 봅시다.

**`my_awesome_project/main.py`**
```python
import requests

def get_my_ip():
    try:
        response = requests.get("https://httpbin.org/ip")
        response.raise_for_status()  # 오류가 발생하면 예외를 일으킴
        ip_data = response.json()
        print(f"My public IP address is: {ip_data['origin']}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_my_ip()
```

이제 터미널에서 아래 명령어로 실행합니다.

```bash
poetry run python my_awesome_project/main.py
```

**결과:**
```
My public IP address is: [여러분의 IP 주소]
```

### 핵심 명령어 요약

| 명령어 | 설명 |
| :--- | :--- |
| `poetry new <project-name>` | 새로운 파이썬 프로젝트를 생성합니다. |
| `poetry init` | 기존 폴더를 Poetry 프로젝트로 초기화합니다. |
| `poetry add <package-name>` | 프로젝트에 라이브러리를 추가하고 설치합니다. |
| `poetry add <package-name> --group dev` | 개발용 라이브러리를 추가합니다. |
| `poetry install` | `pyproject.toml`을 기반으로 모든 의존성을 설치합니다. |
| `poetry run <command>` | 프로젝트의 가상 환경에서 명령어를 실행합니다. |
| `poetry env activate` | 프로젝트의 가상 환경을 활성화합니다. |
| `poetry remove <package-name>` | 프로젝트에서 라이브러리를 제거합니다. |
| `poetry show` | 설치된 라이브러리 목록을 보여줍니다. |
