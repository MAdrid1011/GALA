"""ctypes loader for the repository-owned Ramulator 2 C ABI bridge."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any

import yaml

from gala_sim.identity import sha256_file

from .backend import MissingMemoryBackend


def _inspect_configuration(path: Path) -> dict[str, int | str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise MissingMemoryBackend("Ramulator configuration root is not a mapping")
    frontend = document.get("frontend")
    memory = document.get("memory_system")
    if not isinstance(frontend, dict) or frontend.get("impl") != "External":
        raise MissingMemoryBackend("Ramulator configuration does not use External frontend")
    if int(frontend.get("clock_ratio", 0)) != 1:
        raise MissingMemoryBackend("Ramulator frontend clock ratio must be one")
    if not isinstance(memory, dict) or memory.get("impl") != "GenericDRAM":
        raise MissingMemoryBackend("Ramulator configuration does not use GenericDRAM")
    if int(memory.get("clock_ratio", 0)) != 1:
        raise MissingMemoryBackend("Ramulator memory clock ratio must be one")
    controllers = memory.get("controllers")
    if not isinstance(controllers, list) or not controllers:
        raise MissingMemoryBackend("Ramulator configuration has no controllers")
    channel_width: int | None = None
    data_rate: int | None = None
    read_latency: int | None = None
    for controller in controllers:
        if not isinstance(controller, dict) or controller.get("impl") != "LPDDR5":
            raise MissingMemoryBackend("Ramulator controller is not LPDDR5")
        dram = controller.get("dram")
        if not isinstance(dram, dict) or dram.get("impl") != "LPDDR5":
            raise MissingMemoryBackend("Ramulator DRAM is not LPDDR5")
        width = int(dram.get("channel_width", 0))
        timing = dram.get("timing")
        if not isinstance(timing, list) or not timing:
            raise MissingMemoryBackend("Ramulator LPDDR5 timing is absent")
        rate = int(timing[0])
        current_read_latency = int(dram.get("read_latency", 0))
        if current_read_latency <= 0:
            raise MissingMemoryBackend("Ramulator LPDDR5 read latency is absent")
        channel_width = width if channel_width is None else channel_width
        data_rate = rate if data_rate is None else data_rate
        read_latency = (
            current_read_latency if read_latency is None else read_latency
        )
        if (
            width != channel_width
            or rate != data_rate
            or current_read_latency != read_latency
        ):
            raise MissingMemoryBackend("Ramulator controllers are not homogeneous")
    return {
        "channels": len(controllers),
        "channel_width_bits": int(channel_width or 0),
        "data_rate_mtps": int(data_rate or 0),
        "read_latency_cycles": int(read_latency or 0),
    }


class NativeRamulator2Binding:
    """Own one independent native Ramulator 2 simulation instance."""

    def __init__(self, bridge_library: Path, configuration: Path, *,
                 expected_bridge_sha256: str, version: str, commit: str,
                 ramulator_library_sha256: str, build_manifest_sha256: str) -> None:
        self.bridge_library = Path(bridge_library).resolve()
        self.configuration = Path(configuration).resolve()
        if not self.bridge_library.is_file():
            raise MissingMemoryBackend("Ramulator bridge library is unavailable")
        self._configuration_metadata = _inspect_configuration(self.configuration)
        self._library = ctypes.CDLL(str(self.bridge_library))
        self._declare_functions()
        self._handle = self._library.gala_ramulator_create(
            str(self.configuration).encode("utf-8")
        )
        if not self._handle:
            self._raise_native("cannot create Ramulator instance")
        native_version = self._library.gala_ramulator_version().decode("utf-8")
        if native_version != version:
            self.close()
            raise MissingMemoryBackend("Ramulator bridge version disagrees with build manifest")
        self._version = native_version
        self._commit = commit
        transaction_bytes = self._library.gala_ramulator_transaction_bytes(self._handle)
        if transaction_bytes <= 0:
            self._raise_native("invalid Ramulator transaction size")
        self._transaction_bytes = int(transaction_bytes)
        self._expected_bridge_sha256 = expected_bridge_sha256
        self._ramulator_library_sha256 = ramulator_library_sha256
        self._build_manifest_sha256 = build_manifest_sha256

    @classmethod
    def from_build_manifest(cls, manifest_path: Path, configuration: Path) -> "NativeRamulator2Binding":
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != (
            "gala-ramulator2-bridge-build-v1"
        ):
            raise MissingMemoryBackend("Ramulator bridge build manifest is invalid")
        try:
            bridge_library = Path(str(manifest["bridge_library"]))
            bridge_sha256 = str(manifest["bridge_library_sha256"])
            ramulator_library = Path(str(manifest["ramulator_library"]))
            ramulator_sha256 = str(manifest["ramulator_library_sha256"])
            if not bridge_library.is_file() or not ramulator_library.is_file():
                raise MissingMemoryBackend("Ramulator build libraries are unavailable")
            return cls(
                bridge_library, configuration,
                expected_bridge_sha256=bridge_sha256,
                version=str(manifest["ramulator_version"]),
                commit=str(manifest["ramulator_commit"]),
                ramulator_library_sha256=ramulator_sha256,
                build_manifest_sha256=sha256_file(manifest_path),
            )
        except (KeyError, OSError) as error:
            raise MissingMemoryBackend(
                "Ramulator bridge build manifest is incomplete"
            ) from error

    def metadata(self) -> dict[str, Any]:
        return {
            "implementation": "Ramulator 2",
            "version": self._version,
            "commit": self._commit,
            "config_sha256": sha256_file(self.configuration),
            "bridge_sha256": self._expected_bridge_sha256,
            "ramulator_library_sha256": self._ramulator_library_sha256,
            "build_manifest_sha256": self._build_manifest_sha256,
            "transaction_bytes": self._transaction_bytes,
            **self._configuration_metadata,
        }

    def try_issue(self, address: int, is_write: bool, request_id: int) -> bool:
        result = self._library.gala_ramulator_try_issue(
            self._handle, int(address), int(bool(is_write)), int(request_id)
        )
        if result < 0:
            self._raise_native("Ramulator request issue failed")
        return result == 1

    def tick(self) -> None:
        if self._library.gala_ramulator_tick(self._handle) != 0:
            self._raise_native("Ramulator tick failed")

    def drain_completions(self) -> tuple[int, ...]:
        count = self._library.gala_ramulator_completion_count(self._handle)
        failure = ctypes.c_size_t(-1).value
        if count == failure:
            self._raise_native("Ramulator completion count failed")
        if count == 0:
            return ()
        output = (ctypes.c_uint64 * count)()
        drained = self._library.gala_ramulator_drain(self._handle, output, count)
        if drained == failure or drained != count:
            self._raise_native("Ramulator completion drain failed")
        return tuple(int(output[index]) for index in range(count))

    def clone(self) -> "NativeRamulator2Binding":
        return type(self)(
            self.bridge_library, self.configuration,
            expected_bridge_sha256=self._expected_bridge_sha256,
            version=self._version,
            commit=self._commit,
            ramulator_library_sha256=self._ramulator_library_sha256,
            build_manifest_sha256=self._build_manifest_sha256,
        )

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._library.gala_ramulator_destroy(handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def _declare_functions(self) -> None:
        library = self._library
        library.gala_ramulator_create.argtypes = [ctypes.c_char_p]
        library.gala_ramulator_create.restype = ctypes.c_void_p
        library.gala_ramulator_destroy.argtypes = [ctypes.c_void_p]
        library.gala_ramulator_destroy.restype = None
        library.gala_ramulator_try_issue.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64,
        ]
        library.gala_ramulator_try_issue.restype = ctypes.c_int
        library.gala_ramulator_tick.argtypes = [ctypes.c_void_p]
        library.gala_ramulator_tick.restype = ctypes.c_int
        library.gala_ramulator_completion_count.argtypes = [ctypes.c_void_p]
        library.gala_ramulator_completion_count.restype = ctypes.c_size_t
        library.gala_ramulator_drain.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ]
        library.gala_ramulator_drain.restype = ctypes.c_size_t
        library.gala_ramulator_transaction_bytes.argtypes = [ctypes.c_void_p]
        library.gala_ramulator_transaction_bytes.restype = ctypes.c_int
        library.gala_ramulator_version.argtypes = []
        library.gala_ramulator_version.restype = ctypes.c_char_p
        library.gala_ramulator_last_error.argtypes = []
        library.gala_ramulator_last_error.restype = ctypes.c_char_p

    def _raise_native(self, prefix: str) -> None:
        detail = self._library.gala_ramulator_last_error().decode("utf-8")
        raise MissingMemoryBackend(f"{prefix}: {detail}")
