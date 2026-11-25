// Embeds assets/logo.ico into leonida-builder.exe so Explorer, the taskbar and shortcuts show the icon.
fn main() {
    println!("cargo:rerun-if-changed=../assets/logo.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../assets/logo.ico");
        res.set("ProductName", "Leonida Builder");
        res.set("FileDescription", "Leonida Builder");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the icon: {e}");
        }
    }
}
