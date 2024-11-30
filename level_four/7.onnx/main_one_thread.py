from mainwindow import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from predict import predict


class UiMain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(UiMain, self).__init__(parent)
        self.setupUi(self)
        self.fileBtn.clicked.connect(self.loadImage)


    # 打开文件功能
    def loadImage(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, '请选择图片','.','图像文件(*.jpg *.jpeg *.png)')
        if self.fname:
            print(self.fname)
            self.Infolabel.setText("文件打开成功\n"+self.fname)
            jpg = QtGui.QPixmap(self.fname).scaled(self.Imglabel.width(),
                                                   self.Imglabel.height())

            self.Imglabel.setPixmap(jpg)

            result = predict(self.fname)
            self.Infolabel.setText(result)

        else:
            # print("打开文件失败")
            self.Infolabel.setText("打开文件失败")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UiMain()
    ui.show()
    sys.exit(app.exec_())
