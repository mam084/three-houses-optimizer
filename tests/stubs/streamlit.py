"""
Minimal Streamlit stub used only by tests/test_app_smoke.py.

Real Streamlit widgets need a live browser session and can't run under
`unittest`; this stand-in reproduces just enough of the API surface
(selectbox/slider/checkbox/multiselect/button/columns/container/expander/
tabs/session_state/cache_data/plotly_chart/...) to import app.py and call
its render_* functions directly, so at least "does this code run without
crashing, with these widget values" is covered by an automated test. It is
NOT a substitute for actually running `streamlit run app.py` - visual
layout, reactivity, and real widget behavior aren't exercised at all, only
the Python logic behind each render function. See test_app_smoke.py for
how WIDGET_OVERRIDES is used to control what each widget "returns".
"""

class _SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


session_state = _SessionState()
WIDGET_OVERRIDES = {}  # key/label -> forced return value, set by the test before calling into app.py
PLOTLY_CALLS = []  # (key, fig) for every plotly_chart() call this run - cleared per-test, see test_app_smoke.py
query_params = {}  # plain dict stand-in for real Streamlit's st.query_params mapping - see app.query_param_str


def _override(key, default):
    return WIDGET_OVERRIDES.get(key, default)


def set_page_config(**kwargs):
    pass


def cache_data(func=None, **kwargs):
    if func is None:
        return lambda f: f
    return func


def title(x):
    pass


def subheader(x):
    pass


CAPTION_CALLS = []  # every caption() call's text this run - cleared per-test, see test_app_smoke.py


def caption(x):
    CAPTION_CALLS.append(x)


def divider():
    pass


def write(x):
    pass


MARKDOWN_CALLS = []  # every markdown() call's raw HTML/text this run - cleared per-test, see test_app_smoke.py


def markdown(x, **kw):
    MARKDOWN_CALLS.append(x)


def info(x):
    pass


WARNING_CALLS = []  # every warning() call's text this run - cleared per-test, see test_app_smoke.py


def warning(x, icon=None):
    WARNING_CALLS.append(x)
    print("st.warning:", x)


def success(x, icon=None):
    print("st.success:", x)


def error(x):
    print("st.error:", x)


def stop():
    raise SystemExit("st.stop() called")


IMAGE_CALLS = []  # first positional arg of every image() call this run - cleared per-test, see test_app_smoke.py


def image(*a, **k):
    if a:
        IMAGE_CALLS.append(a[0])


class _Block:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        # Mirrors real Streamlit: a column/container/tab supports the same
        # widget calls as the top-level `st` module (st.columns()[0].button(...)).
        fn = globals().get(name)
        if fn is None:
            raise AttributeError(name)
        return fn


def columns(spec):
    n = spec if isinstance(spec, int) else len(spec)
    return [_Block() for _ in range(n)]


def container(border=None):
    return _Block()


def expander(label):
    return _Block()


def tabs(labels):
    return [_Block() for _ in labels]


SELECTBOX_CALLS = []  # (key or label, options list) for every selectbox() call this run - see test_app_smoke.py


def selectbox(label, options, index=0, key=None, format_func=None, help=None):
    SELECTBOX_CALLS.append((key or label, list(options)))
    default = options[index] if options else None
    return _override(key or label, default)


def radio(label, options, index=0, key=None, format_func=None, help=None, horizontal=False):
    SELECTBOX_CALLS.append((key or label, list(options)))
    default = options[index] if options else None
    return _override(key or label, default)


def slider(label, min_value=None, max_value=None, value=None, key=None):
    return _override(key or label, value)


def checkbox(label, value=False, key=None, help=None):
    return _override(key or label, value)


def multiselect(label, options, format_func=None, key=None, help=None):
    return _override(key or label, [])


def button(label, type=None, help=None, key=None, on_click=None, args=None, kwargs=None):
    clicked = _override(key or label, False)
    if clicked and on_click is not None:
        on_click(*(args or ()), **(kwargs or {}))
    return clicked


def rerun():
    # Real Streamlit aborts the current script run and restarts it; the
    # stub has no script-run loop to restart, so this is just a no-op -
    # callers that check session_state right after triggering a rerun
    # (rather than relying on the rerun itself) still see correct state.
    pass


def plotly_chart(fig, use_container_width=True, key=None):
    PLOTLY_CALLS.append((key, fig))
