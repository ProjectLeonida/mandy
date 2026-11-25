// Embeds assets/logo.ico into leonida-launcher.exe so Explorer, the taskbar and shortcuts show the icon.
fn main() {
    println!("cargo:rerun-if-changed=../assets/logo.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../assets/logo.ico");
        res.set("ProductName", "Leonida");
        res.set("FileDescription", "Leonida");
        res.compile().expect("could not embed assets/logo.ico");
    }
}
