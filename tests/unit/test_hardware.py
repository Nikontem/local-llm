import subprocess

from local_llm.estimate import GIB
from local_llm.hardware import Machine, Probe, detect, total_ram


def fake_run(outputs):
    def run(args, **kwargs):
        stdout = outputs.get(tuple(args[:2]), "")
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")
    return run


def mac_probe():
    displays = (
        "Graphics/Displays:\n    Apple M4 Pro:\n      Chipset Model: Apple M4 Pro\n"
        "      Total Number of Cores: 20\n"
    )
    outputs = {
        ("sysctl", "-n"): "Apple M4 Pro\n",
        ("system_profiler", "SPDisplaysDataType"): displays,
    }
    return Probe(system="Darwin", machine="arm64", run=fake_run(outputs), which=lambda n: None,
                 read_text=lambda p: "", total_ram=lambda: 48 * GIB)


def test_apple_silicon_is_unified_memory():
    m = detect(reserve_gb=10, probe=mac_probe())
    assert m.gpu_kind == "apple" and m.chip == "Apple M4 Pro" and m.gpu_cores == 20
    assert m.total_ram == 48 * GIB and m.gpu_vram == 48 * GIB
    assert m.budget == 38 * GIB and m.gpu_budget == 38 * GIB
    assert m.describe() == (
        "Apple M4 Pro, 20 GPU cores, 48 GB unified memory"
        " → 38 GB usable for models (10 GB reserved)"
    )
    assert m.fit(20 * GIB) == "comfortable"
    assert m.fit_label(30 * GIB) == "fits" and m.fit_label(40 * GIB) == "too big"


def test_linux_with_nvidia():
    outputs = {("nvidia-smi", "--query-gpu=memory.total"): "24576\n"}
    probe = Probe(system="Linux", machine="x86_64", run=fake_run(outputs),
                  which=lambda n: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None,
                  read_text=lambda p: "processor\t: 0\nmodel name\t: AMD Ryzen 9 7950X\n",
                  total_ram=lambda: 64 * GIB)
    m = detect(reserve_gb=8, probe=probe)
    assert m.gpu_kind == "nvidia" and m.gpu_vram == 24 * GIB and m.chip == "AMD Ryzen 9 7950X"
    assert m.budget == 56 * GIB and m.gpu_budget == 23 * GIB
    assert m.fit_label(20 * GIB) == "fits on GPU"
    assert m.fit_label(30 * GIB) == "fits with CPU offload (slower)"
    assert m.fit_label(60 * GIB) == "too big"
    assert "NVIDIA GPU with 24 GB" in m.describe() and "23 GB on the GPU" in m.describe()


def test_linux_with_amd_json():
    rocm = '{"card0": {"VRAM Total Memory (B)": "17163091968", "VRAM Total Used Memory (B)": "1"}}'
    outputs = {("rocm-smi", "--showmeminfo"): rocm}
    probe = Probe(system="Linux", machine="x86_64", run=fake_run(outputs),
                  which=lambda n: "/opt/rocm/bin/rocm-smi" if n == "rocm-smi" else None,
                  read_text=lambda p: "", total_ram=lambda: 32 * GIB)
    m = detect(probe=probe)
    assert m.gpu_kind == "amd" and m.gpu_vram == 17163091968


def test_cpu_only_and_failures_degrade_gracefully():
    def failing_run(args, **kwargs):
        raise OSError("no such tool")

    def failing_read(path):
        raise OSError("no /proc")

    probe = Probe(system="Linux", machine="aarch64", run=failing_run, which=lambda n: None,
                  read_text=failing_read, total_ram=lambda: 16 * GIB)
    m = detect(probe=probe)
    assert m.gpu_kind == "none" and m.chip == "" and m.gpu_budget == 0
    assert m.budget == 6 * GIB
    assert "CPU only" in m.describe() and m.describe().startswith("aarch64, 16 GB RAM")


def test_machine_is_plain_data():
    m = Machine("Darwin", "arm64", "", GIB, "none", 0, 0, 0)
    assert m.budget == GIB and m.fit(GIB) == "fits"


def test_total_ram_positive():
    assert total_ram() > 0
