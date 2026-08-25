"""Production-safe browser regression for the restricted Field 2 review UI."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
import yaml

from chickpea_ssl.field2_readiness import sha256
from scripts.run_field2_point_annotator_v2 import build_store, handler_factory


PROJECT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT / "configs/field2_chickpea_review_policy.yaml"


def test_restricted_browser_save_resume_and_controls_do_not_touch_production(tmp_path):
    selenium = pytest.importorskip("selenium")
    if not shutil.which("firefox") or not shutil.which("geckodriver"):
        pytest.skip("Firefox/geckodriver browser fixture is unavailable")
    policy = yaml.safe_load(POLICY_PATH.read_text())
    contract_path = PROJECT / policy["outputs"]["contract"]
    if not contract_path.is_file():
        pytest.skip("machine-local frozen review policy is not present")
    v2 = yaml.safe_load((PROJECT / policy["v2_sampling_config"]).read_text())
    production = PROJECT / v2["point_annotation"]["output_root"] / "field2_blind_main_point_annotations.json"
    production_before = sha256(production)
    annotation_root = tmp_path / "isolated-annotations"
    store, spectra, _ = build_store(PROJECT, v2, policy, annotation_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(store, spectra, policy))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    from selenium.webdriver import Firefox
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.support.select import Select
    from selenium.webdriver.support.ui import WebDriverWait

    options = Options()
    options.add_argument("-headless")
    driver = None
    try:
        driver = Firefox(
            options=options,
            service=Service(executable_path=shutil.which("geckodriver"), log_output=str(tmp_path / "geckodriver.log")),
        )
        wait = WebDriverWait(driver, 30)
        driver.set_window_size(1440, 1000)
        driver.get(url)
        wait.until(lambda item: "1 / 477 queue" in item.find_element(By.ID, "counter").text)
        with urlopen(f"{url}/api/samples") as response:
            samples = json.load(response)
        assert len(samples) == 477
        assert {item["domain_name"] for item in samples} <= {"research_crop_area", "unassigned_valid_support"}
        with pytest.raises(HTTPError) as reserve_error:
            urlopen(f"{url}/api/reserve")
        assert reserve_error.value.code == 404

        first_id = driver.execute_script("return window.__field2ViewerDebug.getAnnotation().sample_id")
        label = driver.find_element(By.CSS_SELECTOR, '#labelButtons button[data-value="weed_unspecified"]')
        label.click()
        subtype = wait.until(lambda item: item.find_element(By.ID, "weedSubtype"))
        Select(subtype).select_by_value("tall_grass_weed")
        driver.find_element(By.CSS_SELECTOR, '#confidenceButtons button[data-value="high"]').click()
        note = driver.find_element(By.ID, "note")
        note.send_keys("isolated browser regression")
        driver.find_element(By.ID, "boundaryCorrection").click()
        driver.find_element(By.ID, "saveContinue").click()
        wait.until(lambda _: json.loads(store.json_path.read_text())["revision"] == 2)
        saved = json.loads(store.json_path.read_text())["annotations"][first_id]
        assert saved["selected_label"] == "weed_unspecified"
        assert saved["manual_decision"] == "weed_unspecified"
        assert saved["reference_provenance"] == "investigator_nonchickpea_vegetation"
        assert saved["optional_weed_subtype"] == "tall_grass_weed"
        assert saved["confidence"] == "high" and saved["reviewed"] is True
        assert saved["boundary_needs_correction"] is True

        driver.refresh()
        wait.until(lambda item: "1/477 reviewed" in item.find_element(By.ID, "counter").text)
        assert driver.execute_script("return window.__field2ViewerDebug.getAnnotation().sample_id") != first_id
        assert driver.execute_script("return window.__field2ViewerDebug.eventTargets().controlsShieldPointerEvents") == "none"
        driver.find_element(By.ID, "clear").click()
        wait.until(lambda item: item.switch_to.alert).dismiss()
    finally:
        if driver is not None:
            driver.quit()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert sha256(production) == production_before
