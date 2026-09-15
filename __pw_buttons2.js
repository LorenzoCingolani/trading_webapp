const { chromium } = require('playwright');

async function waitIdle(page) {
  try { await page.waitForSelector('text=RUNNING', { state: 'detached', timeout: 90000 }); } catch (e) {}
  await page.waitForTimeout(800);
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1200 } });
  await page.goto('http://localhost:8599', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForSelector('text=Trading Analytics', { timeout: 20000 });
  await waitIdle(page);
  await page.getByRole('button', { name: 'Main Analysis', exact: true }).click();
  await waitIdle(page);
  await page.waitForSelector('text=Strategies to run', { timeout: 20000 });
  await waitIdle(page);
  await page.locator('text=Run Analysis').first().scrollIntoViewIfNeeded();
  await waitIdle(page);
  await page.screenshot({ path: '__pw_btn2_run_analysis.png', fullPage: false });
  await browser.close();
})().catch((e) => { console.error('SCRIPT_FAILED:', e); process.exit(1); });
