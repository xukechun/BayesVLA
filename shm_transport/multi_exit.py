# -*- coding: utf-8 -*-
#
# Copyright (C) 2018-2019 KuraLabs S.R.L
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
atexit replacement that supports multiprocessing.
"""

import os
import sys
import signal
from logging import getLogger
from collections import OrderedDict
from multiprocessing import current_process


__version__ = '1.5.0'


log = getLogger(__name__)


_MAIN_PROC = None
_REGISTRY = OrderedDict()
_SHARED_REGISTRY = []


def _header():
    pid = os.getpid()
    path = list(_REGISTRY.keys())

    if not path or path[-1] != pid:
        path.append(pid)

    return 'Process "{}" (pid: {}, path: {})'.format(
        current_process().name,
        pid,
        ' -> '.join(map(str, path)),
    )


def run_exitfuncs(exitcode):

    # Run process specific exit functions
    owned = _REGISTRY.get(os.getpid())
    if owned:
        # Reversing the queue allows execute exit functions in LIFO order
        # which is the order that fits most applications. The user should be
        # able to configure this somehow someday.
        #
        # Maybe in install() passing an ordering function? But with functions
        # is no very useful.
        #
        # Passing a weight? But then what about non-weighted function that must
        # retain order. Food for thought.
        try:
            for func in reversed(owned):
                log.debug('{} calling *owned* exit function {} ...'.format(
                    _header(),
                    func,
                ))
                func()
        except Exception:
            log.exception('Exit function {} failed:'.format(func))

    # Run shared exit functions
    try:
        for func in reversed(_SHARED_REGISTRY):
            log.debug('{} calling *shared* exit function {} ...'.format(
                _header(),
                func,
            ))
            func()
    except Exception:
        log.exception('Exit function {} failed:'.format(func))

    # Do exit
    if _MAIN_PROC == os.getpid():
        log.debug('{} system exit'.format(
            _header(),
        ))
        sys.exit(exitcode)

    log.debug('{} subprocess exit'.format(
        _header(),
    ))
    os._exit(exitcode)


def handler(signum, frame):
    log.debug('{} got signal {}'.format(
        _header(),
        signal.Signals(signum).name),
    )

    # Filter SIGINT on subprocesses, so that the parent process drives the
    # shutdown
    if signum == signal.SIGINT and _MAIN_PROC != os.getpid():
        log.debug('{} will ignore SIGINT'.format(_header()))
        return

    run_exitfuncs(signum)


def multiexit_except_hook(exctype, value, traceback):
    """
    Unhandled exception hook that works in a multiprocess environment.
    """
    log.critical(
        'Uncaught exception',
        exc_info=(exctype, value, traceback)
    )
    run_exitfuncs(1)


def multiexit_exit_hook():
    """
    Interpreter shutdown / exit hook.
    """
    # Make sure to the exit hook on standard exit only
    log.info('Interpreter shutting down')
    run_exitfuncs(0)


def install(
    signals=(signal.SIGTERM, signal.SIGINT),
    except_hook=True,
):
    global _MAIN_PROC

    # Check that install ins't called twice or more
    if _MAIN_PROC is not None:
        raise RuntimeError(
            'Please call install() only once in the main process. '
            'Main process already set to process with PID {}.'.format(
                _MAIN_PROC,
            )
        )

    # Check signal handlers sanity
    for signum in signals:
        current_handler = signal.getsignal(signum)

        if all([
            signum == signal.SIGINT,
            current_handler == signal.default_int_handler
        ]):
            continue

        if current_handler in [
            signal.SIG_DFL,
            signal.SIG_IGN,
        ]:
            continue

        raise RuntimeError(
            'multiexit doesn\'t support custom signal handlers for {} '
            'yet. PRs welcome. Current signal handler set to: {}'.format(
                signal.Signals(signum).name,
                current_handler,
            )
        )

    _MAIN_PROC = os.getpid()
    for signum in signals:
        signal.signal(signum, handler)

    # Change system exception hook
    if except_hook:
        sys._excepthook_original = sys.excepthook
        sys.excepthook = multiexit_except_hook


def register(func, shared=False):
    assert callable(func)

    # Check that the handler is installed
    current_handler = signal.getsignal(signal.SIGTERM)
    if current_handler != handler:
        raise RuntimeError(
            'multiexit signal handler isn\'t installed. '
            'Unknown signal handler {}. '
            'Please call install() once on the main process before starting '
            'any subprocess.'.format(
                current_handler,
            )
        )

    if shared:
        process_registry = _SHARED_REGISTRY
    else:
        process_registry = _REGISTRY.setdefault(os.getpid(), [])

    if func not in process_registry:
        process_registry.append(func)
        log.debug('{} added {}exit callable {}'.format(
            _header(),
            'shared ' if shared else '',
            func,
        ))

    return func


def unregister(func):
    assert callable(func)

    process_registry = _REGISTRY.get(os.getpid(), None)
    if process_registry is not None:
        if func in process_registry:
            process_registry.remove(func)
            log.debug('{} removed exit callable {}'.format(
                _header(),
                func,
            ))
            return True

    return False


__all__ = [
    'run_exitfuncs',
    'install',
    'register',
    'unregister',
]
