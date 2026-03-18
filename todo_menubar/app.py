"""Native macOS menu bar wrapper for the local todo overlay."""

from __future__ import annotations

import argparse
import logging
import threading
import time
import urllib.error
import urllib.request

import objc
from AppKit import (
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSFloatingWindowLevel,
    NSMakeRect,
    NSMenu,
    NSMenuItem,
    NSPanel,
    NSStatusBar,
    NSVariableStatusItemLength,
    NSViewHeightSizable,
    NSViewWidthSizable,
    NSWindowCollectionBehaviorMoveToActiveSpace,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSObject, NSURL
from WebKit import WKWebView, WKWebViewConfiguration

from todo_overlay.server import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_URL, STATIC_DIR, create_server


APP_NAME = "HomeAssist Todos"
DEFAULT_WIDTH = 460
DEFAULT_HEIGHT = 700

logger = logging.getLogger(__name__)


class OverlayRuntime:
    """Ensures the overlay server is reachable for the menubar web view."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.server = None
        self.server_thread: threading.Thread | None = None

    def ensure_ready(self) -> str:
        if self._healthy():
            return self.base_url

        self._start_embedded_server()
        if not self._wait_for_health():
            raise RuntimeError("Todo overlay server failed to become ready.")
        return self.base_url

    def shutdown(self) -> None:
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        if self.server_thread is not None:
            self.server_thread.join(timeout=2)
        self.server = None
        self.server_thread = None

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/health", timeout=1.5) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False

    def _start_embedded_server(self) -> None:
        if self.server is not None:
            return

        self.server = create_server(host=self.host, port=self.port)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        logger.info("Started embedded todo overlay server on %s", self.base_url)

    def _wait_for_health(self) -> bool:
        for _ in range(20):
            if self._healthy():
                return True
            time.sleep(0.15)
        return False


class TodoMenubarApp(NSObject):
    """Dockless macOS app with a menu bar button and toggleable web view window."""

    def initWithHost_port_(self, host: str, port: int):  # noqa: N802
        self = objc.super(TodoMenubarApp, self).init()
        if self is None:
            return None

        self.host = host
        self.port = port
        self.runtime = OverlayRuntime(host=host, port=port)
        self.overlay_url = DEFAULT_URL if (host, port) == (DEFAULT_HOST, DEFAULT_PORT) else f"http://{host}:{port}"
        self.status_item = None
        self.window = None
        self.webview = None
        return self

    def applicationDidFinishLaunching_(self, notification):  # noqa: N802
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        self._build_main_menu()
        self._build_status_item()
        self._build_window()
        self._load_overlay()

    def applicationWillTerminate_(self, notification):  # noqa: N802
        self.runtime.shutdown()

    def applicationShouldTerminateAfterLastWindowClosed_(self, application):  # noqa: N802
        return False

    def windowShouldClose_(self, sender):  # noqa: N802
        self.hideWindow_(None)
        return False

    def toggleWindow_(self, sender):  # noqa: N802
        if self.window is not None and self.window.isVisible():
            self.hideWindow_(sender)
        else:
            self.showWindow_(sender)

    def showWindow_(self, sender):  # noqa: N802
        if self.window is None:
            return
        self._load_overlay()
        self._position_window()
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    def hideWindow_(self, sender):  # noqa: N802
        if self.window is not None:
            self.window.orderOut_(None)

    def reloadOverlay_(self, sender):  # noqa: N802
        self._load_overlay(force_reload=True)

    def openInBrowser_(self, sender):  # noqa: N802
        try:
            self.runtime.ensure_ready()
            urllib.request.urlopen(self.overlay_url, timeout=1.0).close()
        except Exception:
            pass
        import webbrowser

        webbrowser.open(self.overlay_url)

    def _build_main_menu(self) -> None:
        main_menu = NSMenu.alloc().init()
        app_menu_item = NSMenuItem.alloc().init()
        main_menu.addItem_(app_menu_item)

        app_menu = NSMenu.alloc().init()
        toggle_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Toggle Todos", "toggleWindow:", "t")
        toggle_item.setTarget_(self)
        app_menu.addItem_(toggle_item)

        reload_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Reload Todos", "reloadOverlay:", "r")
        reload_item.setTarget_(self)
        app_menu.addItem_(reload_item)

        browser_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Open In Browser", "openInBrowser:", "b")
        browser_item.setTarget_(self)
        app_menu.addItem_(browser_item)

        app_menu.addItem_(NSMenuItem.separatorItem())

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(f"Quit {APP_NAME}", "terminate:", "q")
        app_menu.addItem_(quit_item)

        app_menu_item.setSubmenu_(app_menu)
        NSApp.setMainMenu_(main_menu)

    def _build_status_item(self) -> None:
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        button = self.status_item.button()
        button.setTitle_("Todos")
        button.setToolTip_("Toggle HomeAssist todo window")
        button.setTarget_(self)
        button.setAction_("toggleWindow:")

    def _build_window(self) -> None:
        style_mask = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        self.window = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, DEFAULT_WIDTH, DEFAULT_HEIGHT),
            style_mask,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setReleasedWhenClosed_(False)
        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setCollectionBehavior_(NSWindowCollectionBehaviorMoveToActiveSpace)
        self.window.setMovableByWindowBackground_(True)
        self.window.setDelegate_(self)

        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setHasShadow_(True)

        wk_config = WKWebViewConfiguration.alloc().init()
        wk_config.preferences().setValue_forKey_(True, "javaScriptEnabled")
        self.webview = WKWebView.alloc().initWithFrame_configuration_(
            self.window.contentView().bounds(), wk_config
        )
        if self.webview is None:
            logger.error("WKWebView failed to initialize — falling back to basic init")
            self.webview = WKWebView.alloc().init()
        self.webview.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        self.webview.setValue_forKey_(False, "drawsBackground")
        self.window.contentView().addSubview_(self.webview)

    def _load_overlay(self, force_reload: bool = False) -> None:
        if self.webview is None:
            logger.error("Cannot load overlay — webview is not initialized")
            return
        try:
            base_url = self.runtime.ensure_ready()
            self.overlay_url = base_url

            html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
            html = html.replace('href="/styles.css"', 'href="styles.css"')
            html = html.replace('src="/app.js"', 'src="app.js"')
            api_tag = f'<script>window.API_BASE="{base_url}";</script>'
            html = html.replace("</head>", f"{api_tag}\n</head>", 1)

            tmp = STATIC_DIR / "_menubar_index.html"
            tmp.write_text(html, encoding="utf-8")

            file_url = NSURL.fileURLWithPath_(str(tmp))
            dir_url = NSURL.fileURLWithPath_(str(STATIC_DIR))
            self.webview.loadFileURL_allowingReadAccessToURL_(file_url, dir_url)
        except Exception as exc:
            logger.error("Unable to load todo overlay: %s", exc)

    def _position_window(self) -> None:
        if self.window is None or self.status_item is None:
            return

        button = self.status_item.button()
        if button is None or button.window() is None:
            return

        button_frame = button.window().convertRectToScreen_(button.frame())
        width = self.window.frame().size.width
        height = self.window.frame().size.height
        x = button_frame.origin.x + button_frame.size.width - width
        y = button_frame.origin.y - height - 8
        if x < 12:
            x = 12
        if y < 24:
            y = 24
        self.window.setFrame_display_animate_(NSMakeRect(x, y, width, height), True, True)


def run_app(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    app = NSApplication.sharedApplication()
    delegate = TodoMenubarApp.alloc().initWithHost_port_(host, port)
    app.setDelegate_(delegate)
    app.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="macOS menu bar app for HomeAssist todos")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Todo overlay host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Todo overlay port.")
    args = parser.parse_args()
    run_app(host=args.host, port=args.port)
