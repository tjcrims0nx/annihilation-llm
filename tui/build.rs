// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025-2026  grimxlock + contributors (Annihilate fork)

//! Derives the user-facing version from `pyproject.toml` at compile time.
//!
//! The version used to be a string literal in `app.rs`, which silently drifted
//! two releases behind the Python package. `pyproject.toml` is the single place
//! a version is typed by hand; the Python side already reads it back through
//! `importlib.metadata`, and this baking it into the binary gives the TUI the
//! same property.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set by Cargo");
    let pyproject = PathBuf::from(&manifest_dir)
        .parent()
        .expect("tui/ always has a parent directory")
        .join("pyproject.toml");

    // A bump to pyproject.toml has to trigger a rebuild, otherwise the baked-in
    // value goes stale exactly the way the old literal did.
    println!("cargo::rerun-if-changed={}", pyproject.display());

    let contents = fs::read_to_string(&pyproject).unwrap_or_else(|e| {
        panic!(
            "failed to read {} to derive the version: {e}",
            pyproject.display()
        )
    });

    let version = parse_project_version(&contents).unwrap_or_else(|| {
        panic!(
            "no `version` key found in the [project] table of {}",
            pyproject.display()
        )
    });

    // Cargo cannot read pyproject.toml, so the crate version is still typed by
    // hand. Fail loudly on disagreement rather than shipping two versions.
    let crate_version =
        env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION is always set by Cargo");
    assert!(
        crate_version == version,
        "version mismatch: tui/Cargo.toml says {crate_version}, but {} says {version}. \
         Update tui/Cargo.toml to match.",
        pyproject.display()
    );

    println!("cargo::rustc-env=ANNIHILATE_VERSION={version}");
}

/// Extracts the `version` value from the `[project]` table.
///
/// Hand-rolled rather than pulling in the `toml` crate: that would add a
/// build-dependency and lockfile churn for a single lookup, and the TUI builds
/// offline today.
fn parse_project_version(contents: &str) -> Option<String> {
    let mut in_project = false;

    for line in contents.lines() {
        let line = line.trim();

        if line.starts_with('[') {
            // Any other table header ends [project]; `[project.urls]` and
            // friends are separate tables and must not be searched.
            in_project = line == "[project]";
            continue;
        }

        if !in_project {
            continue;
        }

        let Some(value) = line.strip_prefix("version") else {
            continue;
        };
        let Some(value) = value.trim_start().strip_prefix('=') else {
            // Guards against a key like `version_scheme = ...`.
            continue;
        };

        let value = value.trim();
        let quote = value.chars().next()?;
        if quote != '"' && quote != '\'' {
            return None;
        }

        return value[1..].split(quote).next().map(str::to_owned);
    }

    None
}
