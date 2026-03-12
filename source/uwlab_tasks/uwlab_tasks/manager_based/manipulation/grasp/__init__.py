# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

try:
    from .config import franka_leap  # noqa: F401
except (ModuleNotFoundError, ImportError):
    pass
