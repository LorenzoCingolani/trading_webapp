const { chromium } = require('playwright');

async function waitIdle(page) {
  // Wait for Streamlit's running indicator to disappear, then settle.
  try {
    await page.waitForSelector('text=RUNNING', { state: 'detached', timeout: 15000 });
  } catch (e) { /* may not have been showing */ }
  await page.waitForTimeout(800);
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1300 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push('console: ' + msg.text()); });

  await page.goto('http://localhost:8599', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForSelector('text=Trading Analytics', { timeout: 20000 });
  await waitIdle(page);

  await page.getByRole('button', { name: 'Main Analysis', exact: true }).click();
  await waitIdle(page);
  await page.waitForSelector('text=Active instruments', { timeout: 20000 });
  await waitIdle(page);
  await page.screenshot({ path: '__pw2_main_analysis.png', fullPage: true });

  await page.getByRole('button', { name: 'Next >' }).click();
  await waitIdle(page);
  await page.screenshot({ path: '__pw2_after_next.png', fullPage: true });

  await page.getByRole('button', { name: '< Back' }).click();
  await waitIdle(page);
  await page.screenshot({ path: '__pw2_after_back.png', fullPage: true });

  await page.getByRole('button', { name: 'PDM', exact: true }).click();
  await waitIdle(page);
  await page.waitForSelector('text=What this step computes', { timeout: 20000 });
  await waitIdle(page);
  await page.screenshot({ path: '__pw2_pdm.png', fullPage: true });

  console.log('ERRORS_FOUND:' + JSON.stringify(errors));
  await browser.close();
})().catch((e) => { console.error('SCRIPT_FAILED:', e); process.exit(1); });
