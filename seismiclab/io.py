from __future__ import annotations

import pathlib
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# Layout follows codes/segy/header.m (big endian SU headers).
TRACE_HEADER_DTYPE = np.dtype(
    [
        ("tracl", ">i4"),
        ("tracr", ">i4"),
        ("fldr", ">i4"),
        ("tracf", ">i4"),
        ("ep", ">i4"),
        ("cdp", ">i4"),
        ("cdpt", ">i4"),
        ("trid", ">i2"),
        ("nva", ">i2"),
        ("nhs", ">i2"),
        ("duse", ">i2"),
        ("offset", ">i4"),
        ("gelev", ">i4"),
        ("selev", ">i4"),
        ("sdepth", ">i4"),
        ("gdel", ">i4"),
        ("sdel", ">i4"),
        ("swdep", ">i4"),
        ("gwdep", ">i4"),
        ("scalel", ">i2"),
        ("scalco", ">i2"),
        ("sx", ">i4"),
        ("sy", ">i4"),
        ("gx", ">i4"),
        ("gy", ">i4"),
        ("counit", ">i2"),
        ("wevel", ">i2"),
        ("swevel", ">i2"),
        ("sut", ">i2"),
        ("gut", ">i2"),
        ("sstat", ">i2"),
        ("gstat", ">i2"),
        ("tstat", ">i2"),
        ("laga", ">i2"),
        ("lagb", ">i2"),
        ("delrt", ">i2"),
        ("muts", ">i2"),
        ("mute", ">i2"),
        ("ns", ">i2"),
        ("dt", ">i2"),
        ("gain", ">i2"),
        ("igc", ">i2"),
        ("igi", ">i2"),
        ("corr", ">i2"),
        ("sfs", ">i2"),
        ("sfe", ">i2"),
        ("slen", ">i2"),
        ("styp", ">i2"),
        ("stas", ">i2"),
        ("stae", ">i2"),
        ("tatyp", ">i2"),
        ("afilf", ">i2"),
        ("afils", ">i2"),
        ("nofilf", ">i2"),
        ("nofils", ">i2"),
        ("lcf", ">i2"),
        ("hcf", ">i2"),
        ("lcs", ">i2"),
        ("hcs", ">i2"),
        ("year", ">i2"),
        ("day", ">i2"),
        ("hour", ">i2"),
        ("minute", ">i2"),
        ("sec", ">i2"),
        ("timbas", ">i2"),
        ("trwf", ">i2"),
        ("grnors", ">i2"),
        ("grnofr", ">i2"),
        ("grnlof", ">i2"),
        ("gaps", ">i2"),
        ("otrav", ">i2"),
        ("d1", ">f4"),
        ("f1", ">f4"),
        ("d2", ">f4"),
        ("f2", ">f4"),
        ("ungpow", ">f4"),
        ("unscale", ">f4"),
        ("ntr", ">i4"),
        ("mark", ">i2"),
        ("unass", ">i2", (15,)),
    ]
)


def read_su(path: str | pathlib.Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Minimal SU reader mirroring ``readsegy.m``.

    Parameters
    ----------
    path:
        Path to a big-endian SU/SEGY file.

    Returns
    -------
    data:
        ``(nt, ntraces)`` float32 array.
    headers:
        List of structured numpy headers matching ``TRACE_HEADER_DTYPE``.
    """

    file_path = pathlib.Path(path)
    traces: List[np.ndarray] = []
    headers: List[np.ndarray] = []

    with file_path.open("rb") as f:
        while True:
            raw_hdr = f.read(TRACE_HEADER_DTYPE.itemsize)
            if len(raw_hdr) < TRACE_HEADER_DTYPE.itemsize:
                break
            header = np.frombuffer(raw_hdr, dtype=TRACE_HEADER_DTYPE, count=1)[0].copy()
            ns = int(header["ns"])
            if ns <= 0:
                break
            trace = np.fromfile(f, dtype=">f4", count=ns)
            if trace.size != ns:
                break
            traces.append(trace.astype(np.float32))
            headers.append(header)

    if not traces:
        raise ValueError(f"No traces found in {path}")

    data = np.stack(traces, axis=1)
    return data, headers


def write_su(
    path: str | pathlib.Path,
    data: np.ndarray,
    headers: Optional[Sequence[np.ndarray]] = None,
    dt: Optional[float] = None,
) -> None:
    """
    Write a matrix of traces to SU on disk.

    Parameters
    ----------
    path:
        Destination file path.
    data:
        ``(nt, ntraces)`` array of float data.
    headers:
        Optional list of numpy headers. If omitted, simple headers are built
        with sequential trace numbers.
    dt:
        Sampling interval in seconds. Only used when ``headers`` are not
        provided.
    """

    file_path = pathlib.Path(path)
    data = np.asarray(data, dtype=np.float32)
    nt, ntraces = data.shape

    with file_path.open("wb") as f:
        for idx in range(ntraces):
            if headers:
                header = headers[idx].copy()
            else:
                header = np.zeros((), dtype=TRACE_HEADER_DTYPE)
                header["tracl"] = idx + 1
                header["tracr"] = idx + 1

            header["ns"] = nt
            if dt is not None:
                header["dt"] = int(round(dt * 1_000_000))
            f.write(header.tobytes())
            traces_bytes = np.asarray(data[:, idx], dtype=">f4").tobytes()
            f.write(traces_bytes)


def extract_header_word(
    path: str | pathlib.Path, hw: str
) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Extract a header word from a SU-SEGY file.

    转换自 MATLAB: codes/segy/extract.m

    Parameters
    ----------
    path : str or pathlib.Path
        Path to SU file
    hw : str
        Header word to extract. Examples:
        - 'cdp': CDP numbers
        - 'offset': offset values
        - 'sx': source X coordinate
        - 'gx': geophone X coordinate
        - 'trace': extract traces (returns data, headers)

    Returns
    -------
    value : np.ndarray or tuple
        Extracted header values, or (data, headers) if hw='trace'

    Examples
    --------
    >>> cdp = extract_header_word('data.su', 'cdp')
    >>> offset = extract_header_word('data.su', 'offset')
    >>> d, h = extract_header_word('data.su', 'trace')
    """
    if hw == 'trace':
        return read_su(path)

    data, headers = read_su(path)
    ntraces = len(headers)

    # Map header word names to numpy dtype field names
    hw_map = {
        'tracl': 'tracl',
        'tracr': 'tracr',
        'fldr': 'fldr',
        'tracf': 'tracf',
        'ep': 'ep',
        'cdp': 'cdp',
        'cdpt': 'cdpt',
        'trid': 'trid',
        'nva': 'nva',
        'nhs': 'nhs',
        'duse': 'duse',
        'offset': 'offset',
        'gelev': 'gelev',
        'selev': 'selev',
        'sdepth': 'sdepth',
        'gdel': 'gdel',
        'sdel': 'sdel',
        'swdep': 'swdep',
        'gwdep': 'gwdep',
        'scalel': 'scalel',
        'scalco': 'scalco',
        'sx': 'sx',
        'sy': 'sy',
        'gx': 'gx',
        'gy': 'gy',
        'counit': 'counit',
        'wevel': 'wevel',
        'swevel': 'swevel',
        'sut': 'sut',
        'gut': 'gut',
        'sstat': 'sstat',
        'gstat': 'gstat',
        'tstat': 'tstat',
        'laga': 'laga',
        'lagb': 'lagb',
        'delrt': 'delrt',
        'muts': 'muts',
        'mute': 'mute',
        'ns': 'ns',
        'dt': 'dt',
        'gain': 'gain',
        'igc': 'igc',
        'igi': 'igi',
        'corr': 'corr',
        'sfs': 'sfs',
        'sfe': 'sfe',
        'slen': 'slen',
        'styp': 'styp',
        'stas': 'stas',
        'stae': 'stae',
        'tatyp': 'tatyp',
        'afilf': 'afilf',
        'afils': 'afils',
        'nofilf': 'nofilf',
        'nofils': 'nofils',
        'lcf': 'lcf',
        'hcf': 'hcf',
        'lcs': 'lcs',
        'hcs': 'hcs',
        'year': 'year',
        'day': 'day',
        'hour': 'hour',
        'minute': 'minute',
        'sec': 'sec',
        'timbas': 'timbas',
        'trwf': 'trwf',
        'grnors': 'grnors',
        'grnofr': 'grnofr',
        'grnlof': 'grnlof',
        'gaps': 'gaps',
        'otrav': 'otrav',
        'd1': 'd1',
        'f1': 'f1',
        'd2': 'd2',
        'f2': 'f2',
        'ungpow': 'ungpow',
        'unscale': 'unscale',
        'ntr': 'ntr',
        'mark': 'mark',
    }

    if hw not in hw_map:
        raise ValueError(f"Unknown header word: {hw}")

    field_name = hw_map[hw]
    values = np.array([h[field_name] for h in headers])

    return values


def sort_su(
    data: np.ndarray, headers: List[np.ndarray], hw: str = 'cdp'
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Sort data and headers using a header word.

    转换自 MATLAB: codes/segy/ssort.m

    Parameters
    ----------
    data : np.ndarray
        Data matrix (nt, ntraces)
    headers : list of np.ndarray
        List of trace headers
    hw : str, optional
        Header word to sort by (default: 'cdp')
        Options: 'cdp', 'offset', 'sx', 'gx', 'sy', 'gy', 'tracl', 'tracr'

    Returns
    -------
    data_sorted : np.ndarray
        Sorted data
    headers_sorted : list of np.ndarray
        Sorted headers

    Notes
    -----
    This is a simple implementation for small to medium datasets.
    Not recommended for very large datasets.

    Examples
    --------
    >>> d, h = read_su('data.su')
    >>> d_sorted, h_sorted = sort_su(d, h, 'cdp')
    """
    hw_map = {
        'tracl': 'tracl',
        'tracr': 'tracr',
        'fldr': 'fldr',
        'tracf': 'tracf',
        'ep': 'ep',
        'cdp': 'cdp',
        'offset': 'offset',
        'sx': 'sx',
        'sy': 'sy',
        'gx': 'gx',
        'gy': 'gy',
    }

    if hw not in hw_map:
        raise ValueError(f"Unknown sort key: {hw}")

    field_name = hw_map[hw]
    sort_values = np.array([h[field_name] for h in headers])

    # Get sort indices
    sort_idx = np.argsort(sort_values)

    # Sort data and headers
    data_sorted = data[:, sort_idx]
    headers_sorted = [headers[i] for i in sort_idx]

    return data_sorted, headers_sorted
