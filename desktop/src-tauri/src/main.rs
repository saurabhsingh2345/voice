// Local Voice Agent desktop shell.
//
// The whole UI already exists as a local web app, so this shell does two jobs:
// start the Python server as a child process, and show its page in a native
// window. Everything still runs on this machine; the window points at
// 127.0.0.1 and nothing is fetched from the internet.
//
// The server is a child process rather than a bundled runtime, which means the
// app currently requires the project checkout and `uv` to be present. Embedding
// a Python runtime is a separate job -- and one with a licensing constraint:
// num2words is LGPL, so the dependency must stay replaceable rather than being
// frozen into one opaque archive.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

const HOST: &str = "127.0.0.1";
const PORT: u16 = 8823;
/// Model loading is lazy, so the server binds quickly. This is generous anyway
/// because a cold `uv run` may resolve the environment first.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(90);

/// Holds the server process so it can be killed when the window closes.
/// Without this the Python process outlives the app and keeps the port bound --
/// observed in practice, so the child is reaped from both the window-close
/// event and the app exit event. Whichever fires first wins; `take()` makes the
/// second call a no-op.
struct ServerProcess(Mutex<Option<Child>>);

impl ServerProcess {
    fn reap(&self) {
        if let Some(mut child) = self.0.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn project_root() -> Option<PathBuf> {
    // In a bundled .app the binary sits in Contents/MacOS, so walk up looking
    // for the project's pyproject.toml. During `tauri dev` the CWD is already
    // inside the project.
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        candidates.extend(exe.ancestors().map(PathBuf::from));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.extend(cwd.ancestors().map(PathBuf::from));
    }
    candidates
        .into_iter()
        .find(|dir| dir.join("pyproject.toml").is_file())
}

fn port_is_open() -> bool {
    TcpStream::connect_timeout(
        &format!("{HOST}:{PORT}").parse().expect("valid socket address"),
        Duration::from_millis(300),
    )
    .is_ok()
}

/// Where to look for `uv` when the venv is missing.
///
/// A GUI-launched app inherits a minimal PATH (`/usr/bin:/bin:/usr/sbin:/sbin`),
/// not the login shell's. `uv` lives in /opt/homebrew/bin, so `Command::new("uv")`
/// fails when the app is double-clicked even though it works from a terminal --
/// which is exactly the case that matters here.
const UV_CANDIDATES: &[&str] = &[
    "/opt/homebrew/bin/uv",
    "/usr/local/bin/uv",
    ".local/bin/uv",
    ".cargo/bin/uv",
];

fn find_uv() -> Option<PathBuf> {
    for candidate in UV_CANDIDATES {
        let path = if candidate.starts_with('/') {
            PathBuf::from(candidate)
        } else {
            std::env::var_os("HOME").map(PathBuf::from)?.join(candidate)
        };
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn log_file(root: &PathBuf) -> Stdio {
    // Silencing the child made a failed launch indistinguishable from a slow
    // one. Keep the output so there is something to read.
    std::fs::File::create(root.join("data").join("server.log"))
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null())
}

fn spawn_server(root: &PathBuf) -> std::io::Result<Child> {
    let _ = std::fs::create_dir_all(root.join("data"));

    // Prefer the venv's own entry point: it needs neither uv nor PATH.
    let direct = root.join(".venv").join("bin").join("voice-web");
    if direct.is_file() {
        return Command::new(direct)
            .current_dir(root)
            .stdout(log_file(root))
            .stderr(log_file(root))
            .spawn();
    }

    // Otherwise fall back to uv, resolved by absolute path.
    let uv = find_uv().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "neither .venv/bin/voice-web nor uv was found; run `uv sync` in the project",
        )
    })?;
    Command::new(uv)
        .args(["run", "voice-web"])
        .current_dir(root)
        .stdout(log_file(root))
        .stderr(log_file(root))
        .spawn()
}

fn wait_for_server() -> bool {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    while Instant::now() < deadline {
        if port_is_open() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    false
}

fn main() {
    let app = tauri::Builder::default()
        .manage(ServerProcess(Mutex::new(None)))
        .setup(|app| {
            // Reuse a server the user already started rather than fighting it
            // for the port.
            let already_running = port_is_open();

            if !already_running {
                match project_root() {
                    Some(root) => match spawn_server(&root) {
                        Ok(child) => {
                            *app.state::<ServerProcess>().0.lock().unwrap() = Some(child);
                        }
                        Err(err) => {
                            eprintln!("could not start the voice server: {err}");
                        }
                    },
                    None => eprintln!("could not locate the project (no pyproject.toml above the binary)"),
                }
            }

            let url = if wait_for_server() {
                format!("http://{HOST}:{PORT}")
                    .parse()
                    .map(WebviewUrl::External)
                    .unwrap_or(WebviewUrl::App("index.html".into()))
            } else {
                // Fall back to the bundled page, which explains what went wrong
                // instead of showing a blank window.
                WebviewUrl::App("index.html".into())
            };

            let window = WebviewWindowBuilder::new(app, "main", url)
                .title("Local Voice Agent")
                .inner_size(860.0, 940.0)
                .min_inner_size(520.0, 600.0)
                .build()?;

            // Closing the window is the common way to quit, and on macOS it can
            // tear the process down before the app-level exit event is handled.
            let handle = app.handle().clone();
            window.on_window_event(move |event| {
                if matches!(event, tauri::WindowEvent::Destroyed | tauri::WindowEvent::CloseRequested { .. }) {
                    handle.state::<ServerProcess>().reap();
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the app");

    app.run(|handle, event| {
        if let RunEvent::ExitRequested { .. } | RunEvent::Exit = event {
            handle.state::<ServerProcess>().reap();
        }
    });
}
