// Embeds assets/logo.ico into leonida-launcher.exe so Explorer, the taskbar and shortcuts show the icon.
// The target is asked, not the host, so a Windows build made on Linux or macOS gets it too (that needs llvm-rc for
// MSVC targets, or a MinGW windres).
fn main() {
    println!("cargo:rerun-if-changed=../assets/logo.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let mut res = winresource::WindowsResource::new();
    res.set_icon("../assets/logo.ico");
    res.set("ProductName", "Leonida");
    res.set("FileDescription", "Leonida");
    res.compile().expect("could not embed assets/logo.ico");
}
