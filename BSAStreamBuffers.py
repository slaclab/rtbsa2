"""
objects for streaming BSA data in real time
"""
__version__ = '0.1'
__author__ = 'Zack Buschmann <zack@slac.stanford.edu>'


from numpy import nan, array, ndarray, floor, sign, roll, vstack
from subprocess import check_output
from threading import Lock
from copy import deepcopy
from warnings import warn

from epics import get_pv


# naming conventions for system edefs
HISTORY_EDEFS = {
    'NC_SXR':  'HSTCUS',
    'NC_HXR':  'HSTCUH',
    'SC_BSYD': 'HSTSCD',
    'SC_SXR':  'HSTSCS',
    'SC_HXR':  'HSTSCH',
    'F2':      'HST',
    }

BEAM_RATE_PVS = {
    'NC_SXR':  'EVNT:SYS0:1:NC_SOFTRATE',
    'NC_HXR':  'EVNT:SYS0:1:NC_HARDRATE',
    'SC_BSYD': 'TPG:SYS0:1:DST02:RATE_RBV',
    'SC_SXR':  'TPG:SYS0:1:DST04:RATE_RBV',
    'SC_HXR':  'TPG:SYS0:1:DST03:RATE_RBV',
    'F2':      'EVNT:SYS1:1:BEAMRATE',
    }
BEAMLINES = HISTORY_EDEFS.keys()
MAX_BUFRATES = {'NC': 120.0, 'SC': 102.0, 'F2': 30.0}
AC_FIDUCIAL_RATE = 360.0  # pulse ID tick rate is 360Hz
BSA_BUFFER_LENGTH = 2800  # don't think this is set to change any time soon...


def ns_to_pulse_ID(ns):
    """ (for BSA PVs) retrieves the pulse ID from the lower 14 bits of the nanoseconds field """
    return int(ns & 0x3fff)

def _push_to_ring_buffer(B, value): B = roll(B, -1); B[-1] = value; return B


class BSAStreamBuffer():
    """
    streams BSA PV data in real time, without monitoring history buffers
    any missing pulses will be filled with NaNs
    note: constructor arguments are also settable attributes

    Args:
        channel: BSA PV channel address (without event definition)
        beamline: which beamline to pull BSA data for

    Attributes:
        sample_rate: BSA buffer event rate (this could fixed, or the actual beam rate)
        sample_spacing: time (s) between samples, equals 1 / sample_rate
        ticks_per_sample: pulse ID increment per buffer update, equals 360Hz / sample_rate
        buffer_modulus: number of beam pulses counted before pulse ID values roll over to 0

    Methods:
        get_data: returns an array of 2800 data points and the corresponding latest pulse ID

    Example usage:
        >> stream = BSAStreamBuffer('BLEN:LI21:265:AIMAX', beamline='NC_HXR')
        >> buffer, pulse_ID = stream.get_data()
    """
    def __init__(self, channel, beamline):
        self._channel, self._beamline = channel, beamline
        self._pv, self._pv_rate, self._pv_history = None, None, None
        self._mutex = Lock()
        self._reinit(raise_errors=True)

    def _reinit(self, raise_errors=False):
        # reinitialize underlying data structures, run automatically on config change
        # raise_errors=False allows (intermediate) undefined channel + beamline combos
        # so the user can reinit like so: s.beamline = 'SC_SXR'; s.channel = 'blahblah'
        try:
            self._buffer = array(BSA_BUFFER_LENGTH)
            if self._pv: self._pv.clear_callbacks()
            self._pv = get_pv(f'{self.channel}', form='time')
            self._pv.wait_for_connection()
            self._pv.clear_callbacks()
            self._pv_rate = get_pv(BEAM_RATE_PVS[self.beamline])
            self._pv_rate.wait_for_connection()
            self._pv_rate.clear_callbacks()
            self._pv_rate.add_callback(self._rate_update)

            # use whatever the fastest-populating edef is for the current beam rate
            self._sample_rate = min(self._pv_rate.get(), MAX_BUFRATES[self.beamline[:2]])
            self._rate_update(value=self._sample_rate)
            suffix = '1H'
            if self._sample_rate >= 10.0:
                suffix = 'TH'
            elif self._sample_rate >= MAX_BUFRATES[self.beamline[:2]]:
                suffix = 'HH' if self.beamline[:2] == 'SC' else 'BR'
            self._pv_history = get_pv(
                f'{self.channel}{HISTORY_EDEFS[self.beamline]}{suffix}', form='time'
                )
            self._pv_history.wait_for_connection()

            # initial population from history buffer, then connect callbacks to start stream
            # mutex (hopefully) makes this happen in quick enough sequence to minimize missed shots
            # TODO: determine if this does anything lol
            with self._mutex:
                v = self._pv_history.get_with_metadata()
                self._p_latest = ns_to_pulse_ID(v['nanoseconds'])
                self._p_prev = self._p_latest - self.ticks_per_sample
                self._buffer = v['value']
                self._pv.add_callback(self._stream)

            self._pv_history.disconnect()

        except Exception as e:
            if raise_errors:
                print(f'{self.beamline} BSAStreamBuffer init for {self.channel} failed')
                raise(e)
            else:
                warn(f'Invalid BSAStreamBuffer definition: {self.beamline} {self.channel}')

    def get_data(self):
        """ return array of 2800 data points and the pulse ID of the latest value """
        return self._buffer, self._p_latest

    def _stream(self, value, nanoseconds, **kw):
        # append the latest value to the stream buffer, if any pulses have been missed
        # since the last update, they are padded with NaNs

        silence = kw.pop("silence", False)
        
        if not self._sample_rate: return
        p_new = ns_to_pulse_ID(nanoseconds)
        b = deepcopy(self._buffer)
        p_expected = (self._p_prev + self.ticks_per_sample) % 2**14
        jump = int((p_new - p_expected) / self.ticks_per_sample)
        if jump > 0:
            if not silence:
                print(f'{self.channel} missed {jump} pulses: {self._p_prev}->{p_new}')
            b = roll(b, -1*jump)
            b[-1*jump:] = nan

        # update to pulse ID and data buffer must be atomic to avoid sync errors
        # otherwise it's possible to get a mismatch if _buffer or _p_latest are updated
        # while get_data has actions on the stack
        # TODO: determine if this does anything lol
        with self._mutex:
            self._buffer = _push_to_ring_buffer(b, value)
            self._p_prev = self._p_latest
            self._p_latest = p_new


    

    def _rate_update(self, value, **kw):
        # updates current buffer sample rate and derived quantities
        self._sample_rate = min(value, MAX_BUFRATES[self.beamline[:2]])
        self._sample_spacing =   nan if not value else 1.0 / self._sample_rate
        self._ticks_per_sample = nan if not value else AC_FIDUCIAL_RATE / self._sample_rate
        self._buffer_modulus =   nan if not value else floor(2**14 / self._ticks_per_sample)

    def stop(self):
        for pv in [self._pv, self._pv_rate]:
            pv.clear_callbacks()
            pv.disconnect()

    @property
    def beamline(self): return self._beamline

    @beamline.setter
    def beamline(self, value):
        if value not in BEAMLINES: raise ValueError(f'{value} is not a valid beamline')
        self._beamline = value
        self._reinit()

    @property
    def channel(self): return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value
        self._reinit()

    @property
    def sample_rate(self): return self._sample_rate

    @property
    def sample_spacing(self): return self._sample_spacing

    @property
    def ticks_per_sample(self): return self._ticks_per_sample

    @property
    def buffer_modulus(self): return self._buffer_modulus


class dualBSAStreamBuffer():
    """
    paired, synchronized array of BSA data for ch1 & ch2
    capable of streaming N <= BSA_BUFFER_LENGTH points of synced data
    streams data with two underlying BSAStreamBuffer objects, synchronizes on requeset
    note: constructor arguments are also settable attributes
    note: shares sample_rate and derived attributes with BSAStreamBuffer

    Args:
        ch1: BSA PV channel address (without event definition)
        ch2: BSA PV channel address (without event definition)
        beamline: which beamline to pull BSA data for

    Attributes:
        N_pts_sync: current number of synchronized data points
 
    Methods:
        get_data: returns an 2xN_pts_sync array and the corresponding latest pulse ID

    Example usage:
        >> stream = dualBSAStreamBuffer(
               'BLEN:LI21:265:AIMAX', 'GDET:FEE1:241:ENRC', beamline='NC_HXR'
               )
        >> buffer, pulse_ID = stream.get_data() 
    """
    def __init__(self, ch1, ch2, beamline):
        self.__doc__ += f'\n{BSAStreamBuffer.__doc__}'
        self._ch1, self._ch2, self._beamline = ch1, ch2, beamline
        self._p_latest, self.N_pts_sync = -1, -1
        self._reinit(raise_errors=True)

    def _reinit(self, raise_errors=False):
        # reinitialize underlying BSAStreamBuffers, run automatically on config change
        try:
            if self._p_latest > 0: self.stop()
            self._s1 = BSAStreamBuffer(self._ch1, self._beamline)
            self._s2 = BSAStreamBuffer(self._ch2, self._beamline)
        except Exception as e:
            if raise_errors:
                print(f'{self.beamline} dualBSAStreamBuffer init with [{self.ch1}, {self.ch2}] failed')
                raise(e)
            else:
                warn('Invalid dualBSAStreamBuffer definition')

    def get_data(self):
        """ returns a 2XM array, (M<=2800) of actual synchronized data & its latest pulse ID """
        (b1,p1), (b2,p2) = self._s1.get_data(), self._s2.get_data()

        dp = p2 - p1
        self._p_latest = min(p1, p2)
        if not dp: return vstack((b1,b2)), self._p_latest # already synced!

        # synchronize buffers b1 and b2 with last withnessed pulse IDs p1 and p2
        # pulse IDs count up to 2^14, check for rollover by
        # comparing the raw shot delta to a shifted one and take the minimum
        # i.e. when the raw delta equals the buffer modulus, the rollover delta is 0
        # lag_direction = sign(dp)
        dshot_raw = int(dp / self.ticks_per_sample)
        dshot_rollover = int(dshot_raw - self.buffer_modulus)
        shot_offset = sign(dp)*min(abs(dshot_raw), abs(dshot_rollover))

        # sync data by shifting buffers +/- by the shot offset
        # shot_offset > 0 means p1 lags p2 and vice-versa
        self.N_pts_sync = BSA_BUFFER_LENGTH - abs(shot_offset)
        b_synced = ndarray((2, self.N_pts_sync))
        if shot_offset > 0:
            b_synced[0,:] = b1[shot_offset:]
            b_synced[1,:] = b2[:self.N_pts_sync]
        elif shot_offset < 0:
            b_synced[0,:] = b1[:self.N_pts_sync]
            b_synced[1,:] = b2[-1*shot_offset:]

        return b_synced, self._p_latest

    def stop(self):
        if self._s1: self._s1.stop()
        if self._s2: self._s2.stop()

    @property
    def beamline(self): return self._beamline

    @beamline.setter
    def beamline(self, value):
        if value not in BEAMLINES: raise ValueError(f'{value} is not a valid beamline')
        self._beamline = value
        self._reinit()

    @property
    def ch1(self): return self._s1.channel

    @ch1.setter
    def ch1(self, value):
        self._s1.channel = value
        self._reinit()

    @property
    def ch2(self): return self._s2.channel

    @ch2.setter
    def ch2(self, value):
        self._s2.channel = value
        self._reinit()

    @property
    def sample_rate(self): return self._s1.sample_rate

    @property
    def sample_spacing(self): return self._s1.sample_spacing

    @property
    def ticks_per_sample(self): return self._s1.ticks_per_sample

    @property
    def buffer_modulus(self): return self._s1.buffer_modulus


#===================================================================================================
# old code  -- this was functional before starting buffer->scalar conversion
# delete when psychological safety blanket is no longer needed
#     @property
#     def B(self):
#         return self.data
#         # return self._get_synced_history() #kludge
#     def _get_synced_history(self):
#         v1, v2 = self._pv1.get_with_metadata(), self._pv2.get_with_metadata()
#         p1, p2 = ns_to_pulse_ID(v1['nanoseconds']), ns_to_pulse_ID(v2['nanoseconds'])
#         raw_buffer = ndarray((2, BSA_BUFFER_LENGTH))
#         raw_buffer[0,:], raw_buffer[1,:] = v1['value'], v2['value']

#         dp = p2 - p1;
#         self.pulse_ID = min(p1, p2)
#         # self._pulse_IDs[0,-1] = self.pulse_ID
#         # self._pulse_IDs[1,-1] = self.pulse_ID
#         if not dp: return raw_buffer

#         # pulse IDs count up to 2^14, check for rollover by
#         # comparing the raw shot delta to a shifted one and take the minimum
#         # i.e. when the raw delta equals the buffer modulus, the rollover delta is 0
#         lag_direction = sign(dp)
#         dshot_raw = int(dp / self.ticks_per_sample)
#         dshot_rollover = int(dshot_raw - self.buffer_modulus)
#         shot_offset = lag_direction*min(abs(dshot_raw), abs(dshot_rollover))

#         # sync data by shifting buffers +/- by the shot offset
#         # shot_offset > 0 means p1 lags p2 and vice-versa
#         self.N_pts_sync = BSA_BUFFER_LENGTH - abs(shot_offset)
#         b_synced = ndarray((2, self.N_pts_sync))
#         if shot_offset > 0:
#             b_synced[0,:] = raw_buffer[0, shot_offset:]
#             b_synced[1,:] = raw_buffer[1, :self.N_pts_sync]
#         elif shot_offset < 0:
#             b_synced[0,:] = raw_buffer[0, :self.N_pts_sync]
#             b_synced[1,:] = raw_buffer[1, -1*shot_offset:]
#         return b_synced

