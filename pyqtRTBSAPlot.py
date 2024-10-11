"""
PyQt object that plots real time BSA data
time series & FFT mode for a single PV, and correlation scatterplot for 2
"""
__version__ = '0.1'
__author__ = 'Zack Buschmann <zack@slac.stanford.edu>'

# TODO: stop plotting when beam rate 0 (also handle divide by 0 errors)
# TODO: add a median filter/moving average to time series plot


from numpy import nanmean, nanstd, isfinite, linspace, polyfit, poly1d
from numpy.fft import fft, fftfreq
from copy import deepcopy
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor
from pyqtgraph import PlotWidget, PlotCurveItem, ScatterPlotItem, LabelItem

from epics import get_pv
from BSAStreamBuffers import BSAStreamBuffer, dualBSAStreamBuffer

# filtering RuntimeWarnings so numpy wont spam stdout with empty slice warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


DEFAULT_DRAW_INTERVAL = 100
BSA_BUFFER_LENGTH = 2800
FIT_LINE_NPTS = 500
QCOL_R = QColor(255,0,0)
QCOL_B = QColor(0,0,255)
ANNOTATION_ALPHA = 0.75
DATA_TEXT_POS = (0.075,0.01)
FIT_TEXT_POS = (0.075,0.04)


class _rtbsaPlot(PlotWidget):
    """
    _rtbsaPlot: parent class of rtbsaCorrPlot, rtbsaTimePlot, not for direct use
    note: constructor arguments are also settable attributes

    Args:
        beamline: which beamline to pull BSA data for
        start_paused: whether to start updating the plot with new data immediately on creation
        N_pts: number of data points to show, must be 1 < N < BSA_BUFFER_LENGTH
        filter_data: flag to filter points that are > N_sigma std deviations from the mean
        N_sigma: integer number of standard deviations for filter cutoff
        annotate: flag to show plot annotations

    Attributes:
        N_pts_actual: number of currently available data points

    Methods:
        restart_update: starts or restarts the real-time plot updates
        stop_update: stops real-time plot updates
    """
    def __init__(self, beamline, start_paused=True, N_pts=BSA_BUFFER_LENGTH,
        filter_data=True, N_sigma=3.0, show_annotation=True, parent=None, **kw
        ):
        PlotWidget.__init__(self, parent=parent, background='w', foreground='k')

        self.beamline = beamline
        self.start_paused = start_paused
        self.N_pts = N_pts
        self.N_pts_actual = -1
        self.filter_data = filter_data
        self.N_sigma = N_sigma
        self.show_annotation = show_annotation

        self._data_text = LabelItem('', color=QCOL_B)
        self._data_text.setParentItem(self.getPlotItem())
        self._data_text.setOpacity(ANNOTATION_ALPHA)
        self._data_text.anchor(itemPos=(0,0), parentPos=DATA_TEXT_POS)
        self.getAxis('bottom').setTextPen('k')
        self.getAxis('left').setTextPen('k')
        self.showGrid(1, 1)

        self._draw_timer = QTimer(self)
        self._draw_timer.setInterval(DEFAULT_DRAW_INTERVAL)
        self._draw_timer.timeout.connect(self._update_plot)
        if not start_paused: self._draw_timer.start()

    def _update_plot(self): pass

    def kill_stream(self): raise NotImplementedError

    def stop_update(self): self._draw_timer.stop()

    def restart_update(self): self.stop_update(); self._draw_timer.start()

    def _filter_outliers(self, B): return abs(B-nanmean(B)) < self.N_sigma*nanstd(B)


class rtbsaCorrPlot(_rtbsaPlot):
    """
    Qt plot widget to display real time beam-synchronous scatter plots
    includes filtering based on mean-std dev & polynomial fitting up to degree 2
    note: constructor arguments are also settable attributes

    Args: 
        ch1: BSA PV channel address (without event definition)
        ch2: BSA PV channel address (without event definition)
        show_fit: flag to show
        fit_order: polynomial degree for data fitting, must be <= 2

    Attributes:
        stream: underlying dualBSAStreamBuffer object
        fit_coeffs: calculated fitting coefficients returned from numpy.polyfit

    Methods:
        get_annotations: returns two strings describing current data and fit status
    """
    def __init__(self, ch1, ch2, show_fit=False, fit_order=1,**kw):
        _rtbsaPlot.__init__(self, **kw)
        self.__doc__ += f'\n{_rtbsaPlot.__doc__}'
        self.fit_order = fit_order
        self.fit_coeffs = None
        self.stream = dualBSAStreamBuffer(ch1, ch2, beamline=self.beamline)
        self._p_latest = -1

        self._scatter = ScatterPlotItem(symbol='o', size=4)
        self._scatter.setBrush(QCOL_B)
        self._scatter.setOpacity(1.0)
        self._show_fit = show_fit
        self._fit_line = PlotCurveItem(pen=QCOL_R)
        self._fit_line.setOpacity(0.0)
        self._fit_text = LabelItem('', color=QCOL_R)
        self._fit_text.setParentItem(self.getPlotItem())
        self._fit_text.setOpacity(ANNOTATION_ALPHA)
        self._fit_text.anchor(itemPos=(0,0), parentPos=FIT_TEXT_POS)
        
        self.getPlotItem().setLabel(axis='bottom', text=self.stream.ch1)
        self.getPlotItem().setLabel(axis='left', text=self.stream.ch2)
        self.addItem(self._scatter)
        self.addItem(self._fit_line)

    def kill_stream(self): self.stream.stop()

    @property
    def show_fit(self): return self._show_fit

    @show_fit.setter
    def show_fit(self, value):
        self._show_fit = value
        alpha = 1.0 if self._show_fit else 0.0
        self._fit_line.setOpacity(alpha)

    def _update_data(self):
        # filters nans (and optionally outlier points) from raw buffer data 
        B, self._p_latest = self.stream.get_data()
        Bx, By = B[0,-1*self.N_pts:], B[1,-1*self.N_pts:]
        mask = isfinite(Bx) & isfinite(By)
        if self.filter_data:
            mask = mask & self._filter_outliers(Bx) & self._filter_outliers(By)
        self.N_pts_actual = sum(mask)
        return Bx[mask], By[mask]

    def _update_plot(self):
        x, y = self._update_data()
        self._scatter.setData(x, y)
        if self.show_fit:
            self.fit_coeffs = polyfit(x, y, self.fit_order)
            yp = poly1d(self.fit_coeffs)
            xp = linspace(min(x), max(x), FIT_LINE_NPTS)
            self._fit_line.setData(xp, yp(xp))
        self._annotate()

    def _annotate(self):
        if not self.show_annotation: return
        datamsg, fitmsg = self.get_annotations()
        self._data_text.setText(datamsg)
        self._fit_text.setText(fitmsg)

    def get_annotations(self):
        """
        Return strings deciribing the current dataset

        Returns:
            datamsg: string describing number of datapoints and pulse ID
            fitmsg: described fit parameters if fitting is enabled, empty string if not
        """
        if not self.N_pts_actual: return 'NO DATA', ''
        datamsg = f'N pts: {self.N_pts_actual}/{self.N_pts}'
        datamsg = datamsg + f', latest pulse ID: {self._p_latest}'
        fitmsg = ''
        if self.show_fit:
            fitmsg = f'O(x^{self.fit_order}) fit'
            for i in range(len(self.fit_coeffs)):
                fitmsg = fitmsg + f',  a{i} = {self.fit_coeffs[i]:.3f}'
        return datamsg, fitmsg


class rtbsaTimePlot(_rtbsaPlot):
    """
    Qt plot widget to display a stream of BSA data, or an fft of the same time series
    note: constructor arguments are also settable attributes

    Args:
        channel: BSA PV channel address (without event definition)
        plot_fft: flag to show the power spectral density of the signal

    Attributes:
        stream: underlying BSAStreamBuffer object

    Methods:
        get_annotation: returns a string describing current data status
    """

    def __init__(self, channel, plot_fft=False, **kw):
        _rtbsaPlot.__init__(self, **kw)
        self.__doc__ += f'\n{_rtbsaPlot.__doc__}'
        self.channel = channel
        self.plot_fft = plot_fft
        self.stream = BSAStreamBuffer(self.channel, self.beamline)

        self._raw_buffer = self.stream.get_data()
        self._t = linspace(0, BSA_BUFFER_LENGTH, BSA_BUFFER_LENGTH)
        self._domain = None
        self._line = PlotCurveItem(pen=QCOL_B, antialias=True)
        self._line.setOpacity(1.0)
        self.addItem(self._line)

    def kill_stream(self): self.stream.stop()

    @property
    def plot_fft(self): return self._plot_fft
    
    @plot_fft.setter
    def plot_fft(self, value):
        self._plot_fft = value
        self.getPlotItem().setLabel(axis='bottom', text='frequency')
        xlabel, ylabel = 'shot index', self.channel
        if self.plot_fft:
            xlabel, ylabel = 'frequency [Hz]', f'power spectral density ({self.channel})'
        self.getPlotItem().setLabel(axis='bottom', text=xlabel)
        self.getPlotItem().setLabel(axis='left', text=ylabel)

    def _update_data(self):
        B, self._p_latest = self.stream.get_data()
        mask = isfinite(B)
        if self.filter_data: mask = mask & self._filter_outliers(B)
        self.N_pts_actual = sum(mask)
        self._raw_buffer = (B[mask])[-1*self.N_pts:]
        tt = self._t[mask]
        self._domain = tt[-1*self.N_pts:]
        # self._domain = (self._t[-1*self.N_pts:])[mask]

    def _update_plot(self):
        self._update_data()
        x, y = self._domain, self._raw_buffer
        if self.plot_fft:
            psd = abs(fft(self._raw_buffer - nanmean(self._raw_buffer))**2)
            f = fftfreq(self.N_pts_actual, self.stream.sample_spacing)
            freq_domain = (f > 0)
            x, y = f[freq_domain], psd[freq_domain]
        self._line.setData(x, y)
        self._annotate()

    def get_annotation(self):
        """ Returns a string describing the number of data points and latest pulse ID"""
        if not self.N_pts_actual: return 'NO DATA'
        datamsg = f'N pts: {self.N_pts_actual}/{self.N_pts}'
        datamsg = datamsg + f', latest pulse ID: {self._p_latest}'
        return datamsg

    def _annotate(self):
        if not self.show_annotation: return
        self._data_text.setText(self.get_annotation())

    