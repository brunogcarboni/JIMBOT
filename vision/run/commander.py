"""Rumo a partir da caixa. O terminal só imprime quando o rótulo muda."""

import serial

import config
from tracker import TrackState

SERIAL_CHAR = {
    "reto": "r",
    "reto-direita": "r",
    "reto-esquerda": "r",
    "esquerda": "e",
    "buscando-esquerda": "e",
    "direita": "d",
    "buscando-direita": "d",
    "parado": "p",
    "perdido": "p",
}


class Commander:
    def __init__(
        self,
        port: str | None = config.SERIAL_PORT,
        baud: int = config.SERIAL_BAUD,
    ) -> None:
        self.last: str | None = None
        self._last_char: str | None = None
        self._is_near = False
        self._serial: serial.Serial | None = None
        if port:
            try:
                self._serial = serial.Serial(port, baud, timeout=0)
            except serial.SerialException as exc:
                raise SystemExit(f"Não abriu a serial {port}: {exc}") from exc

    def update(self, state: TrackState) -> str:
        if not state.visible or state.lost or state.nx is None or state.width_ratio is None:
            if state.exit_side == "direita":
                command = "buscando-direita"
            elif state.exit_side == "esquerda":
                command = "buscando-esquerda"
            else:
                command = "perdido"
        elif self._near(state.width_ratio):
            command = "parado"
        else:
            side = self._side(state.nx)
            if side == "centro":
                command = "reto"
            elif side == "direita":
                command = "reto-direita"
            else:
                command = "reto-esquerda"

        if command != self.last:
            print(command, flush=True)
            self.last = command
        char = SERIAL_CHAR[command]
        if char != self._last_char:
            self._write(char)
            self._last_char = char
        return command

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _write(self, char: str) -> None:
        if self._serial is None:
            return
        self._serial.write(char.encode("ascii"))
        self._serial.flush()

    def _side(self, nx: float) -> str:
        if nx > config.CENTER_DEADZONE:
            return "direita"
        if nx < -config.CENTER_DEADZONE:
            return "esquerda"
        return "centro"

    def _near(self, width_ratio: float) -> bool:
        if self._is_near:
            self._is_near = width_ratio >= config.NEAR_EXIT
        else:
            self._is_near = width_ratio >= config.NEAR_ENTER
        return self._is_near
