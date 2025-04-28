import time
import atexit
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.box import MINIMAL
from rich.text import Text
import plotext as plt
import torch


def generate_table(epoch, loss, val_loss):
    # 設定 padding 為 0 以減少額外空白
    table = Table(title="Training Progress", padding=(0, 0))
    table.add_column("Epoch", justify="right", style="cyan", no_wrap=True)
    table.add_column("Loss", style="magenta")
    table.add_column("Val Loss", style="green")
    table.add_row(str(epoch), f"{loss:.3f}", f"{val_loss:.3f}")
    return table

class LossCurve:
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.colors = "green"
        self.theme = "clear"

    def update(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(loss)

        plt.clt()
        plt.clear_data()
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.theme("clear")
        plt.ticks_color('green')

        plt.plot(self.epochs, self.losses, color=self.colors)
        plt.xfrequency(len(self.epochs) + 1)
        plt.xlim(0, max(self.epochs))
        min_loss, max_loss = min(self.losses), max(self.losses)
        best_epoch = np.argmin(self.losses)
        plt.vline(best_epoch, color="red")
        plt.ylim(min_loss * 0.9, max_loss * 1.1)
        plt.plotsize(50, 15)

    def build(self):
        return plt.build()


if __name__ == "__main__":
    # 建立 Live 並使用 screen=True 強制全螢幕重繪
    live = Live()
    live.start()
    atexit.register(live.stop)

    epochs = []
    losses = []
    val_losses = []

    # 利用 Layout 分割上下區
    layout = Layout(name="root")
    # 上區放表格，下區放 ASCII 圖形，給下區固定高度（例如 25 行）
    layout.split_column(
        Layout(name="upper", size=6),
        Layout(name="lower", size=15)
    )
    plt_mgr = LossCurve()
    for epoch in range(1, 100):
        loss = 1.0 / epoch
        val_loss = 1.2 / epoch

        if epoch > 50:
            val_loss += epoch * 0.01

        plt_mgr.update(epoch, loss)

        # 取得圖形字串並移除末尾空白
        plot_str = plt_mgr.build()

        # 用 Text 建立純文本，不做換行與格式解析
        plot_text = Text(plot_str, overflow="ignore", no_wrap=False) #no_wrap=False, overflow="ignore"
        # 更新 Layout 的兩個區域
        #layout["upper"].update(table)
        layout["lower"].update(plot_text)

        live.update(layout, refresh=True)
        time.sleep(0.1)