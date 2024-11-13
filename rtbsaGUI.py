"""
real-time BSA plotting GUI (2)
capable of displaying time series plots + FFTs and correlation scatterplots
"""
__version__ = '0.1'
__author__ = 'Zack Buschmann <zack@slac.stanford.edu>'


import os
import sys
from subprocess import check_output, CalledProcessError
from socket import gethostname
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QVBoxLayout, QCompleter
from pyqtgraph.exporters import ImageExporter

from physicselog import submit_entry
from meme.names import list_pvs
from pydm import Display
from pyqtRTBSAPlot import rtbsaCorrPlot, rtbsaTimePlot


SELF_PATH = os.path.dirname(os.path.abspath(__file__))

LCLS_DEAFULT_BEAMLINE = 'NC_HXR'
DEFAULT_CHANNELS = {
    'NC_SXR':  ['BLEN:LI21:265:AIMAX', 'EM1K0:GMD:HPS:milliJoulesPerPulse'],
    'NC_HXR':  ['BLEN:LI21:265:AIMAX', 'GDET:FEE1:241:ENRC'],
    'SC_BSYD': ['BPMS:BC1B:125:X', 'BPMS:BC1B:440:X'],
    'SC_SXR':  ['BLEN:BC1B:850:1:BLEN', 'EM1K0:GMD:HPS:milliJoulesPerPulse'],
    'SC_HXR':  ['BLEN:BC1B:850:1:BLEN', 'BLEN:BC2B:950:1:BLEN'],
    'F2':      ['BPMS:IN10:221:X', 'BPMS:IN10:221:TMIT'],
    }
BEAMLINES = DEFAULT_CHANNELS.keys()
BSA_NAMELISTS = {
    'NC_SXR':  'LCLS.CUS.BSA.rootnames',
    'NC_HXR':  'LCLS.CUH.BSA.rootnames',
    'SC_BSYD': 'LCLS.SCB.BSA.rootnames',
    'SC_SXR':  'LCLS.SCS.BSA.rootnames',
    'SC_HXR':  'LCLS.SCH.BSA.rootnames',
    'F2':      'FACET.BSA.rootnames',
    }

DEFAULT_PLOT_MODE = 'corr'

# style sheets for control buttons
TEXT_ON = 'color: rgb(0, 0, 0);'
TEXT_OFF = 'color: rgb(120, 120, 120);'
BG_START = 'background-color: rgb(79, 216, 69);'
BG_STOP = 'background-color: rgb(221, 55, 55);'
BG_LOG = 'background-color: rgb(89, 227, 255);'
STYLE_ROOT_ON = f'{TEXT_ON}\n{{}}'
STYLE_ROOT_OFF = f'{TEXT_OFF}\n{{}}'
STYLE_START_ON = STYLE_ROOT_ON.format(BG_START)
STYLE_START_OFF = STYLE_ROOT_OFF.format(BG_START)
STYLE_STOP_ON = STYLE_ROOT_ON.format(BG_STOP)
STYLE_STOP_OFF = STYLE_ROOT_OFF.format(BG_STOP)
STYLE_CLEAR_ON = TEXT_ON
STYLE_CLEAR_OFF = TEXT_OFF
STYLE_LOG_ON = STYLE_ROOT_ON.format(BG_LOG)
STYLE_LOG_OFF = STYLE_ROOT_OFF.format(BG_LOG)

# facility-dependent colors for the window banner
BANNER_NC = 'background-color: rgb(0, 170, 255);'
BANNER_SC = 'background-color: rgb(255, 55, 95);'
BANNER_F2 = 'background-color: rgb(255, 149, 60);'
BANNERS = {'NC':BANNER_NC, 'SC':BANNER_SC, 'F2':BANNER_F2}

class rtbsaGUI(Display):
    def __init__(self, parent=None, args=None):
        Display.__init__(self, parent=parent, args=args)

        # disable beamline toggles for lcls/facet as needed
        is_F2 = (gethostname() in ['facet-srv01', 'facet-srv02'])
        self.bl_NC_SXR.setEnabled(not is_F2)
        self.bl_NC_HXR.setEnabled(not is_F2)
        self.bl_SC_BSYD.setEnabled(not is_F2)
        self.bl_SC_SXR.setEnabled(not is_F2)
        self.bl_SC_HXR.setEnabled(not is_F2)
        self.bl_F2.setEnabled(is_F2)
        self.bl_F2.setChecked(is_F2)
        if is_F2:
            self.beamline = 'F2'
        else:
            self.beamline = LCLS_DEAFULT_BEAMLINE

        # container for BSA PVs, only populate as needed
        self.bsa_PV_lists = {}
        for bl in BEAMLINES: self.bsa_PV_lists[bl] = None

        self.plotcontainer.setLayout(QVBoxLayout())
        self.plot, self.mode, self.need_reinit = None, DEFAULT_PLOT_MODE, False
        self.init_plot(
            DEFAULT_CHANNELS[self.beamline][0], DEFAULT_CHANNELS[self.beamline][1], self.beamline
            )
        self.set_beamline()

        self.startButton.clicked.connect(self.restart_plot)
        self.stopButton.clicked.connect(self.stop_plot)
        self.clearButton.clicked.connect(self.clear_plot)
        self.physLogButton.clicked.connect(self.log_plot_phys)
        self.mccLogButton.clicked.connect(self.log_plot_mcc)
        for bl_ctrl in [
            self.bl_NC_SXR, self.bl_NC_HXR, self.bl_SC_BSYD,
            self.bl_SC_SXR, self.bl_SC_HXR, self.bl_F2
            ]:
            bl_ctrl.clicked.connect(self.set_beamline)
        for mode_ctrl in [
            self.plotmode_corr,  self.plotmode_time,  self.plotmode_fft
            ]:
            mode_ctrl.clicked.connect(self.set_plotmode)
        self.setNpts.valueChanged.connect(self.set_filtering)
        self.set_filt.clicked.connect(self.set_filtering)
        self.set_Nsigma.valueChanged.connect(self.set_filtering)
        self.set_fit.clicked.connect(self.set_fitting)
        self.sel_fitord.valueChanged.connect(self.set_fitting)

        self.sel_fitord.setValue(1)
        self.toggle_stop_start(True)
        self.toggle_clear_log(False)
        self.status.setText('Hello world!')

    def ui_filename(self): return os.path.join(SELF_PATH, 'rtbsa.ui')

    def set_beamline(self):
        if   self.bl_NC_SXR.isChecked(): self.beamline = 'NC_SXR'
        elif self.bl_NC_HXR.isChecked(): self.beamline = 'NC_HXR'
        elif self.bl_SC_BSYD.isChecked(): self.beamline = 'SC_BSYD'
        elif self.bl_SC_SXR.isChecked(): self.beamline = 'SC_SXR'
        elif self.bl_SC_HXR.isChecked(): self.beamline = 'SC_HXR'
        elif self.bl_F2.isChecked(): self.beamline = 'F2'

        self.banner.setStyleSheet(BANNERS[self.beamline[:2]])
        self.get_BSA_PVs()
        for pvsel in [self.pvsel_1, self.pvsel_2]:
            # disable callbacks while updating dropdown menus
            pvsel.disconnect()
            pvsel.clear()
            pvsel.addItems(self.bsa_PV_lists[self.beamline])
            # comboBox QCompleter settings make dropdown lists searchable
            pvsel.completer().setCompletionMode(QCompleter.PopupCompletion)
            pvsel.completer().setFilterMode(Qt.MatchContains)
            pvsel.activated.connect(self.reinit_plot)
        self.pvsel_1.setCurrentIndex(
            self.bsa_PV_lists[self.beamline].index(DEFAULT_CHANNELS[self.beamline][0])
            )
        self.pvsel_2.setCurrentIndex(
            self.bsa_PV_lists[self.beamline].index(DEFAULT_CHANNELS[self.beamline][1])
            )
        self.reinit_plot()
        self.status.setText(f'Set beamline to: {self.beamline}')

    def get_BSA_PVs(self):
        if not self.bsa_PV_lists[self.beamline]:
            self.status.setText('Updating BSA PV list ...')
            try:
                self.bsa_PV_lists[self.beamline] = list_pvs('%', tag=BSA_NAMELISTS[self.beamline])
            except Exception as e:
                print(repr(e))
                self.status.setText('Failed to get BSA device list. Toggle beamline to retry.')
                self.bsa_PV_lists[self.beamline] = ['']
        return self.bsa_PV_lists[self.beamline]

    def set_plotmode(self):
        is_corr_plot = self.plotmode_corr.isChecked()
        self.pvsel_2.setEnabled(is_corr_plot)
        self.pv2_label.setEnabled(is_corr_plot)
        self.ctrl_fit.setEnabled(is_corr_plot)
        if is_corr_plot:
            self.mode = 'corr'
        elif self.plotmode_time.isChecked():
            self.mode = 'time'
        else:
            self.mode = 'fft'
        self.reinit_plot()

    def init_plot(self, ch1, ch2, beamline):
        if self.plot:
            self.plot.kill_stream()
            self.plot.setParent(None)
        if self.mode == 'corr':
            self.plot = rtbsaCorrPlot(ch1=ch1, ch2=ch2, beamline=beamline, start_paused=True)
        else:
            self.plot = rtbsaTimePlot(channel=ch1, beamline=beamline, plot_fft=(self.mode=='fft'))
        self.plot.N_pts = self.setNpts.value()
        self.ui.plotcontainer.layout().addWidget(self.plot)

    def reinit_plot(self):
        self.stop_plot()
        ch1, ch2 = self.pvsel_1.currentText(), self.pvsel_2.currentText()
        self.init_plot(ch1, ch2, self.beamline)

    def set_filtering(self):
        self.plot.N_pts = self.setNpts.value()
        self.plot.filter_data = self.set_filt.isChecked()
        self.plot.N_sigma = self.set_Nsigma.value()

    def set_fitting(self):
        self.plot.show_fit = self.set_fit.isChecked()
        self.plot.fit_order = self.sel_fitord.value()

    def restart_plot(self):
        if self.need_reinit:
            self.ui.status.setText('Restarting plot ...')
            self.reinit_plot()
            self.need_reinit = False
        self.plot.restart_update()
        self.ui.status.setText('Plot updating ...')
        self.toggle_stop_start(False)
        self.toggle_clear_log(True)

    def stop_plot(self):
        self.plot.stop_update()
        self.ui.status.setText('Plot updating stopped.')
        self.toggle_stop_start(True)

    def clear_plot(self):
        self.stop_plot()
        self.plot.clear()
        self.ui.status.setText('Plot cleared.')
        self.toggle_clear_log(False)
        self.need_reinit = True

    def toggle_stop_start(self, state):
        self.ui.startButton.setEnabled(state)
        self.ui.stopButton.setEnabled(not state)
        self.set_button_styles()

    def toggle_clear_log(self, state):
        self.ui.clearButton.setEnabled(state)
        self.ui.physLogButton.setEnabled(state)
        self.ui.mccLogButton.setEnabled(state)
        self.set_button_styles()

    def set_button_styles(self):
        s_start = STYLE_START_ON if self.ui.startButton.isEnabled() else STYLE_START_OFF
        s_stop = STYLE_STOP_ON if self.ui.stopButton.isEnabled() else STYLE_STOP_OFF
        s_clear = STYLE_CLEAR_ON if self.ui.clearButton.isEnabled() else STYLE_CLEAR_OFF
        s_log = STYLE_LOG_ON if self.ui.physLogButton.isEnabled() else STYLE_LOG_OFF
        self.ui.startButton.setStyleSheet(s_start)
        self.ui.stopButton.setStyleSheet(s_stop)
        self.ui.clearButton.setStyleSheet(s_clear)
        self.ui.physLogButton.setStyleSheet(s_log)
        self.ui.mccLogButton.setStyleSheet(s_log)

    def _save_plot(self): ImageExporter(self.plot.getPlotItem()).export('/tmp/RTBSA.png')

    def log_plot_phys(self):
        self._save_plot()
        facility = self.beamline[:2]
        if   facility == 'NC': logname = 'lcls'
        elif facility == 'SC': logname = 'lcls2'
        else: logname = 'facet'
        if self.mode == 'corr':
            s1, s2 = self.plot.get_annotations()
            logtxt = f'{s1}\n{s2}'
            desc = f'{self.plot.stream.ch1} vs {self.plot.stream.ch2}'
        else:
            d1, d2 = f'{self.plot.stream.channel}', 'vs time'
            if self.mode == 'fft': d2 = 'FFT'
            desc = f'{d1} {d2}'
            logtxt = self.plot.get_annotation()
        
        submit_entry(
            logbook=logname, username='Real-time BSA GUI',
            title=f'{self.beamline} BSA data: {desc}',
            entry_text=logtxt, attachment='/tmp/RTBSA.png'
            )
        self.status.setText(f'Sent to {logname} physics elog')

    def log_plot_mcc(self):
        self._save_plot()
        os.system(f'lpr -P elog_mcc /tmp/RTBSA.png')
        self.status.setText('Sent to MCC E-log')

    
