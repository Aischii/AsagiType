//! Background thread that owns the candle models (loaded on first use) and runs
//! detection + OCR and cleaning on page images.

use eframe::egui;
use std::{
    path::PathBuf,
    sync::mpsc::{channel, Receiver, Sender},
    thread,
};
use vision::{
    clean::{clean_page_full, CleanBox},
    Models, TextRegion,
};

pub enum Request {
    Analyze {
        page: usize,
        bytes: Vec<u8>,
        replace: bool,
    },
    Clean {
        page: usize,
        /// Signature of the box state being cleaned, echoed back.
        sig: u64,
        bytes: Vec<u8>,
        boxes: Vec<CleanBox>,
        /// Balloon id of each box, to report touch-ups.
        ids: Vec<u32>,
    },
}

pub enum Outcome {
    Analyzed {
        replace: bool,
        regions: Vec<TextRegion>,
    },
    Cleaned {
        sig: u64,
        layer: image::RgbaImage,
        mask: image::GrayImage,
        /// Balloon ids that need a manual touch-up.
        touch_up: Vec<u32>,
    },
}

pub struct PageResult {
    pub page: usize,
    pub result: Result<Outcome, String>,
}

pub struct VisionWorker {
    tx: Sender<Request>,
    rx: Receiver<PageResult>,
    pub models_dir: PathBuf,
    pub pending: usize,
}

const THRESHOLD: f32 = 0.5;

fn run(
    models: &mut Option<Models>,
    dir: &std::path::Path,
    req: Request,
) -> (usize, Result<Outcome, String>) {
    match req {
        Request::Analyze {
            page,
            bytes,
            replace,
        } => {
            let result = (|| {
                if models.is_none() {
                    *models = Some(Models::load(dir).map_err(|e| {
                        format!("Could not load models from {}:\n{e:#}", dir.display())
                    })?);
                }
                let img = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
                let regions = models
                    .as_ref()
                    .unwrap()
                    .analyze(&img, THRESHOLD)
                    .map_err(|e| format!("{e:#}"))?;
                Ok(Outcome::Analyzed { replace, regions })
            })();
            (page, result)
        }
        Request::Clean {
            page,
            sig,
            bytes,
            boxes,
            ids,
        } => {
            let result = (|| {
                let img = image::load_from_memory(&bytes)
                    .map_err(|e| e.to_string())?
                    .to_rgb8();
                let res = clean_page_full(&img, &boxes);
                let touch_up = ids
                    .iter()
                    .zip(&res.touch_up)
                    .filter(|(_, t)| **t)
                    .map(|(id, _)| *id)
                    .collect();
                Ok(Outcome::Cleaned {
                    sig,
                    layer: res.layer,
                    mask: res.mask,
                    touch_up,
                })
            })();
            (page, result)
        }
    }
}

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
                    let (page, result) = run(&mut models, &dir, req);
                    let _ = res_tx.send(PageResult { page, result });
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
