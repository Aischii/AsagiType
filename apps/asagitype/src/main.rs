#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod credits;
mod settings;
mod theme;
mod vision_worker;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("AsagiType 2000")
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([900.0, 600.0])
            .with_decorations(false)
            .with_resizable(true),
        ..Default::default()
    };
    eframe::run_native(
        "AsagiType 2000",
        options,
        Box::new(|cc| Ok(Box::new(app::AsagiApp::new(cc)))),
    )
}
