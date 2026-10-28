"""Rótulos discretos. O terminal só recebe linha quando o rótulo muda."""

import config
from tracker import TrackState


class Commander:
    def __init__(self) -> None:
        self.last: str | None = None
        self._is_near = False

    def update(self, state: TrackState) -> str:
        if not state.visible or state.lost or state.nx is None or state.width_ratio is None:
            if state.exit_side == "direita":
                command = "buscando-direita"
            elif state.exit_side == "esquerda":
                command = "buscando-esquerda"
            else:
                command = "perdido"
        else:
            near = self._near(state.width_ratio)
            side = self._side(state.nx)
            if side == "centro":
                command = "parado" if near else "reto"
            elif side == "direita":
                command = "direita" if near else "reto-direita"
            else:
                command = "esquerda" if near else "reto-esquerda"

        if command != self.last:
            print(command, flush=True)
            self.last = command
        return command

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
