# -*- coding: utf-8 -*-
"""桌面 GUI 入口。

用法: python src/app.py
依赖 pywebview（Windows 使用系统自带 WebView2，无需额外安装）。
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _show_error(msg):
    """pythonw 启动无控制台，错误用原生弹窗展示，避免"打不开"无提示。"""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0, f'启动失败：\n\n{msg}', '战锤 PDF 提取工具', 0x10)
    except Exception:
        print(msg)


def main():
    import webview
    from webview.dom import _dnd_state
    from gui_api import GuiApi

    api = GuiApi()
    html = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'gui', 'index.html')

    window = webview.create_window(
        '战锤 PDF 提取工具', html, js_api=api,
        width=1280, height=840, min_size=(1024, 700))
    api.set_window(window)

    def on_drop(_event):
        """pywebview 把拖入的文件路径收集在 _dnd_state['paths']，
        必须注册 drop 监听器才会启用该机制。"""
        dropped = []
        while _dnd_state['paths']:
            _name, path = _dnd_state['paths'].pop(0)
            if path.lower().endswith('.pdf'):
                dropped.append(path)
        if dropped:
            window.evaluate_js(f'addFiles({json.dumps(dropped)});')

    def after_start():
        """窗口打开后注册 DOM 监听（启动前注册会等页面加载超时报错）。

        拖放链路：index.html 内联脚本把 FileList 转数组后直接
        postMessageWithAdditionalObjects('FilesDropped', files) 发给
        Python 侧（pywebview 自带桥未做 FileList 转换，会收空），
        Python 把路径收集到 _dnd_state['paths']，这里取走推回前端。
        """
        from webview.dom import DOMEventHandler
        window.dom.document.events.dragover += DOMEventHandler(
            lambda e: None, True, True)
        window.dom.document.events.drop += DOMEventHandler(
            on_drop, True, True)
        print('DROP_HANDLER_REGISTERED')

    def on_drop(e=None):
        """把拖入的 PDF 路径推给前端 addFiles。"""
        dropped = []
        while _dnd_state['paths']:
            _name, path = _dnd_state['paths'].pop(0)
            if path.lower().endswith('.pdf'):
                dropped.append(path)
        if dropped:
            window.evaluate_js(f'addFiles({json.dumps(dropped)});')

    webview.start(after_start, debug=False)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        _show_error(traceback.format_exc(limit=3))
