""" Callbacks for printing, logging and log information."""
import os
import sys
import time
from contextlib import suppress
from numbers import Number
from itertools import cycle

import numpy as np
import tqdm
from tabulate import tabulate

from enum import Enum

# sys.path.insert(0, os.getcwd())
from .callback_base import Callback

class Ansi(Enum):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'

__all__ = ['EpochTimer', 'PrintLog']


def filter_log_keys(keys, keys_ignored=None):
    """Filter out keys that are generally to be ignored.

    This is used by several callbacks to filter out keys from history
    that should not be logged.

    Parameters
    ----------
    keys : iterable of str
      All keys.

    keys_ignored : iterable of str or None (default=None)
      If not None, collection of extra keys to be ignored.

    """
    keys_ignored = keys_ignored or ()
    for key in keys:
        if not (
                key == 'epoch' or
                (key in keys_ignored) or
                key.endswith('_best') or
                key.endswith('_batch_count') or
                key.startswith('event_')
        ):
            yield key


class EpochTimer(Callback):
    """Measures the duration of each epoch and writes it to the
    history with the name ``dur``.

    """
    def __init__(self, **kwargs):
        super(EpochTimer, self).__init__(**kwargs)

        self.epoch_start_time_ = None

    def on_epoch_begin(self, net, **kwargs):
        self.epoch_start_time_ = time.time()

    def on_epoch_end(self, net, **kwargs):
        net.history.record('dur', time.time() - self.epoch_start_time_)


class PrintLog(Callback):
    """Print useful information from the model's history as a table.

    By default, ``PrintLog`` prints everything from the history except
    for ``'batches'``.

    To determine the best loss, ``PrintLog`` looks for keys that end on
    ``'_best'`` and associates them with the corresponding loss. E.g.,
    ``'train_loss_best'`` will be matched with ``'train_loss'``. The
    :class:`skorch.callbacks.EpochScoring` callback takes care of
    creating those entries, which is why ``PrintLog`` works best in
    conjunction with that callback.

    ``PrintLog`` treats keys with the ``'event_'`` prefix in a special
    way. They are assumed to contain information about occasionally
    occuring events. The ``False`` or ``None`` entries (indicating
    that an event did not occur) are not printed, resulting in empty
    cells in the table, and ``True`` entries are printed with ``+``
    symbol. ``PrintLog`` groups all event columns together and pushes
    them to the right, just before the ``'dur'`` column.

    *Note*: ``PrintLog`` will not result in good outputs if the number
    of columns varies between epochs, e.g. if the valid loss is only
    present on every other epoch.

    Parameters
    ----------
    keys_ignored : str or list of str (default=None)
      Key or list of keys that should not be part of the printed
      table. Note that in addition to the keys provided by the user,
      keys such as those starting with 'event_' or ending on '_best'
      are ignored by default.

    sink : callable (default=print)
      The target that the output string is sent to. By default, the
      output is printed to stdout, but the sink could also be a
      logger, etc.

    tablefmt : str (default='simple')
      The format of the table. See the documentation of the ``tabulate``
      package for more detail. Can be 'plain', 'grid', 'pipe', 'html',
      'latex', among others.

    floatfmt : str (default='.4f')
      The number formatting. See the documentation of the ``tabulate``
      package for more details.

    stralign : str (default='right')
      The alignment of columns with strings. Can be 'left', 'center',
      'right', or ``None`` (disable alignment). Default is 'right' (to
      be consistent with numerical columns).

    """
    def __init__(
            self,
            keys_ignored=None,
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
            stralign='right',
    ):
        self.keys_ignored = keys_ignored
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        self.stralign = stralign

    def initialize(self):
        self.first_iteration_ = True

        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add('batches')
        return self

    def format_row(self, row, key, color):
        """For a given row from the table, format it (i.e. floating
        points and color if applicable).

        """
        value = row[key]

        if isinstance(value, bool) or value is None:
            return '+' if value else ''

        if not isinstance(value, Number):
            return value

        # determine if integer value
        is_integer = float(value).is_integer()
        template = '{}' if is_integer else '{:' + self.floatfmt + '}'

        # if numeric, there could be a 'best' key
        key_best = key + '_best'
        if (key_best in row) and row[key_best]:
            template = color + template + Ansi.ENDC.value
        return template.format(value)

    def _sorted_keys(self, keys):
        """Sort keys, dropping the ones that should be ignored.

        The keys that are in ``self.ignored_keys`` or that end on
        '_best' are dropped. Among the remaining keys:
          * 'epoch' is put first;
          * 'dur' is put last;
          * keys that start with 'event_' are put just before 'dur';
          * all remaining keys are sorted alphabetically.
        """
        sorted_keys = []

        # make sure 'epoch' comes first
        if ('epoch' in keys) and ('epoch' not in self.keys_ignored_):
            sorted_keys.append('epoch')

        # ignore keys like *_best or event_*
        for key in filter_log_keys(sorted(keys), keys_ignored=self.keys_ignored_):
            if key != 'dur':
                sorted_keys.append(key)

        # add event_* keys
        for key in sorted(keys):
            if key.startswith('event_') and (key not in self.keys_ignored_):
                sorted_keys.append(key)

        # make sure 'dur' comes last
        if ('dur' in keys) and ('dur' not in self.keys_ignored_):
            sorted_keys.append('dur')

        return sorted_keys

    def _yield_keys_formatted(self, row):
        colors = cycle([color.value for color in Ansi if color != color.ENDC])
        for key, color in zip(self._sorted_keys(row.keys()), colors):
            formatted = self.format_row(row, key, color=color)
            if key.startswith('event_'):
                key = key[6:]
            yield key, formatted

    def table(self, row):
        headers = []
        formatted = []
        for key, formatted_row in self._yield_keys_formatted(row):
            headers.append(key)
            formatted.append(formatted_row)

        return tabulate(
            [formatted],
            headers=headers,
            tablefmt=self.tablefmt,
            floatfmt=self.floatfmt,
            stralign=self.stralign,
        )

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)

    # pylint: disable=unused-argument
    def on_epoch_end(self, history, verbose, **kwargs):
        data = history[-1]
        tabulated = self.table(data)

        if self.first_iteration_:
            header, lines = tabulated.split('\n', 2)[:2]
            self._sink(header, verbose)
            self._sink(lines, verbose)
            self.first_iteration_ = False

        self._sink(tabulated.rsplit('\n', 1)[-1], verbose)
        if self.sink is print:
            sys.stdout.flush()
