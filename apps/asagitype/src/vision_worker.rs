//! Background thread that owns the candle models (loaded on first use) and
//! runs detection + OCR on page images.

use eframe::egui;
use std::{
    path::PathBuf,
    sync::mpsc::{channel, Receiver, Sender},
    thread,
};
use vision::{Models, TextRegion};

pub struct Request {
    pub page: usize,
    pub bytes: Vec<u8>,
    pub replace: bool,
}

pub struct PageResult {
    pub page: usize,
    pub replace: bool,
    pub result: Result<Vec<TextRegion>, String>,
}

pub struct VisionWorker {
    tx: Sender<Request>,
    rx: Receiver<PageResult>,
    pub models_dir: PathBuf,
    pub pending: usize,
}

const THRESHOLD: f32 = 0.5;

impl VisionWorker {
    pub fn spawn(models_dir: PathBuf, ctx: egui::Context) -> Self {
        let (tx, req_rx) = channel::<Request>();
        let (res_tx, rx) = channel::<PageResult>();
        let dir = models_dir.clone();
        thread::Builder::new()
            .name("asagitype-vision".into())
            .spawn(move || {
                let mut models: Option<Models> = None;
                for req in req_rx {
                    let result = (|| -> Result<Vec<TextRegion>, String> {
                        if models.is_none() {
                            models = Some(Models::load(&dir).map_err(|e| {
                                format!("Could not load models from {}:\n{e:#}", dir.display())
                            })?);
                        }
                        let img = image::load_from_memory(&req.bytes).map_err(|e| e.to_string())?;
                        models
                            .as_ref()
                            .unwrap()
                            .analyze(&img, THRESHOLD)
                            .map_err(|e| format!("{e:#}"))
                    })();
                    let _ = res_tx.send(PageResult {
                        page: req.page,
                        replace: req.replace,
                        result,
                    });
                    ctx.request_repaint();
                }
            })
            .expect("spawn vision thread");
        Self {
            tx,
            rx,
            models_dir,
            pending: 0,
        }
    }

    pub fn submit(&mut self, req: Request) {
        if self.tx.send(req).is_ok() {
            self.pending += 1;
        }
    }

    pub fn poll(&mut self) -> Vec<PageResult> {
        let out: Vec<PageResult> = self.rx.try_iter().collect();
        self.pending = self.pending.saturating_sub(out.len());
        out
    }
}
