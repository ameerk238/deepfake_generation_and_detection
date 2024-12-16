##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, October 2022
##############################################################################
"""doc
# datapipes.utils.decoder

> A generic decoder implementation.

"""
from __future__ import annotations

import io
import os
from typing import Any, Callable

from torch.utils.data.datapipes.datapipe import DataChunk
from torch.utils.data.datapipes.utils.common import StreamWrapper


def _decoder_key_fn(file_path: str) -> str:
    """
    Extract everything after the last but one dot from the file name (path) as a string to be used as the key for
    selecting the proper decoder for the file.

    :param file_path: A full path or filename.
    :return: The key to be used for selecting the proper decoder for the file.
    """
    bn = os.path.basename(file_path)
    key = str.join(".", bn.split(".")[-2:])
    return key


class Decoder:
    def __init__(self,
                 decoders: dict[str, Callable[[str, bytes], Any]],
                 decoder_key_fn: Callable = _decoder_key_fn):
        """ Creates an object that applies the specified decoders to the respective components of samples.

        :param decoders: A dictionary mapping a key to a decoder. By default, the key is the component identifier.
        :param decoder_key_fn: The function that shall be used to extract the decoder key from the file_path of the
         component. Defaults to a function that extracts the component identifier.
        """
        self.decoders = decoders
        self.decoder_key_fn = decoder_key_fn

    @staticmethod
    def _is_stream_handle(data):
        obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
        return isinstance(obj_to_check, io.BufferedIOBase) or isinstance(obj_to_check, io.RawIOBase)

    def _decode(self, sample: DataChunk):
        new_sample = []
        for file_path, data in sample:
            key = self.decoder_key_fn(file_path)
            component, extension = os.path.splitext(key)
            # remove "."
            extension = extension[1:]

            if Decoder._is_stream_handle(data):
                ds = data
                # The behavior of .read can differ between streams (e.g. HTTPResponse), hence this is used instead
                data = b"".join(data)
                ds.close()

            if component in self.decoders.keys():
                new_sample.append((file_path, self.decoders[component](extension, data)))
            else:
                # Pass data on without decoding
                new_sample.append((file_path, data))
        return DataChunk(tuple(new_sample))

    def __call__(self, sample: DataChunk | tuple):
        if isinstance(sample, tuple):
            # handle nested structure when training with sequence data (multiple samples
            parts = [self._decode(p) for p in sample]
            return DataChunk(tuple(parts))
        else:
            # decode a single sample
            return self._decode(sample)
