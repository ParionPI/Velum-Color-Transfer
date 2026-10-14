/* Native Windows save dialog; File System Access fallback on other platforms. */
(() => {
  const button = document.getElementById("btnExportCube");
  let capabilities = null;
  let exporting = false;
  let lastHandle = null;

  function storeHandle(handle, read = false) {
    if (!window.indexedDB) return Promise.resolve(null);
    return new Promise((resolve, reject) => {
      const request = window.indexedDB.open("velum-lut-export", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("destination");
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const tx = db.transaction("destination", read ? "readonly" : "readwrite");
        const query = read ? tx.objectStore("destination").get("last-successful-file") :
          tx.objectStore("destination").put(handle, "last-successful-file");
        tx.oncomplete = () => { db.close(); resolve(read ? query.result : true); };
        tx.onerror = tx.onabort = () => { db.close(); reject(tx.error); };
      };
    });
  }

  async function init() {
    try {
      const response = await fetch("/api/lut/capabilities", {cache: "no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "导出服务不可用");
      capabilities = data;
      if (!data.native_save && typeof window.showSaveFilePicker !== "function") {
        button.title = "此浏览器不支持保存路径选择，请使用本机Windows整合包或Chrome/Edge。";
        return;
      }
      if (!data.native_save) {
        try { lastHandle = await storeHandle(null, true); } catch { /* Browser storage may be disabled. */ }
      }
      button.disabled = false;
    } catch (error) {
      button.title = "导出服务未连接，请重启Velum并刷新网页：" + error.message;
    }
  }

  function successMessage(path, audit, remembered = true) {
    const rmse = Number(audit?.rmse_255);
    const quality = Number.isFinite(rmse) ? `；回放RMSE ${rmse.toFixed(2)}/255` : "";
    const warning = rmse > 4 ? "，空间效果/复杂映射只能近似，请预览检查" : "";
    flash(`已导出：${path}${quality}${warning}${remembered ? "" : "；目录记忆未保存"}`, rmse > 4 || !remembered);
  }

  button.addEventListener("click", async () => {
    if (exporting || !capabilities) return;
    if (!hasBothImages() || !Number.isInteger(serverImageEpoch)) {
      flash("请先上传源图与参考图，再导出LUT。", true); return;
    }
    if (busy || uploadBusy || uploadDirty.size || layerBusy || layerDirty || pendingApply) {
      flash("请等待当前图片和图层处理完成，再导出LUT。", true); return;
    }
    const snapshot = JSON.stringify({params: paramsPayload(), layers: layerPayload(),
      side: Number(resVal()), epoch: serverImageEpoch, size: 65});
    exporting = true;
    button.disabled = true;
    button.textContent = "导出中…";
    let writer = null;
    try {
      // Invoke the browser picker before any await, preserving user activation.
      const handlePromise = capabilities.native_save ? null : window.showSaveFilePicker({
        id: "velum-cube-export", suggestedName: "Velum-Look.cube",
        startIn: lastHandle || "documents", excludeAcceptAllOption: true,
        types: [{description: "3D LUT (.cube)", accept: {"application/octet-stream": [".cube"]}}]
      });
      const handle = handlePromise ? await handlePromise : null;
      if (handle && !handle.name.toLowerCase().endsWith(".cube")) throw new Error("请选择.cube文件名。");
      const response = await fetch(capabilities.native_save ? "/api/lut/save" : "/api/lut/cube", {
        method: "POST", headers: {"Content-Type": "application/json", "X-Velum-Export": capabilities.token},
        body: snapshot
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "LUT导出失败");
      }
      if (capabilities.native_save) {
        const result = await response.json();
        if (result.cancelled) return;
        successMessage(result.path, result.audit, result.directory_remembered);
      } else {
        const content = await response.blob();
        const audit = JSON.parse(response.headers.get("X-Velum-LUT-Audit") || "{}");
        writer = await handle.createWritable();
        await writer.write(content);
        await writer.close();
        writer = null;
        lastHandle = handle;
        let remembered = false;
        try { remembered = !!(await storeHandle(handle)); } catch { /* Saving the LUT itself succeeded. */ }
        successMessage(handle.name, audit, remembered);
      }
    } catch (error) {
      if (writer) { try { await writer.abort(); } catch { /* Preserve the original failure. */ } }
      if (error.name !== "AbortError") {
        if (error.name === "NotFoundError" || error.name === "TypeError") lastHandle = null;
        flash("导出LUT失败：" + error.message, true);
      }
    } finally {
      exporting = false;
      button.disabled = false;
      button.textContent = "导出.cube";
    }
  });
  init();
})();
