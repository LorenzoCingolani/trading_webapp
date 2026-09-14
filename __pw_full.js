const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1200 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push('console: ' + msg.text()); });

  await page.goto('http://localhost:8599', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForSelector('text=Trading Analytics', { timeout: 20000 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: '__pw_nav_settings.png' });

  // Click Main Analysis via top nav button
  await page.getByRole('button', { name: 'Main Analysis', exact: true }).click();
  await page.waitForSelector('text=Active instruments', { timeout: 20000 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: '__pw_nav_main_analysis.png', fullPage: true });

  // Click "Next >" bottom nav button to go to Validation
  await page.getByRole('button', { name: 'Next >' }).click();
  await page.waitForTimeout(1000);
  await page.screenshot({ path: '__pw_nav_after_next.png', fullPage: true });

  // Click "< Back" to return
  await page.getByRole('button', { name: '< Back' }).click();
  await page.waitForTimeout(1000);
  await page.screenshot({ path: '__pw_nav_after_back.png', fullPage: true });

  // Go to PDM page directly via top nav
  await page.getByRole('button', { name: 'PDM', exact: true }).click();
  await page.waitForSelector('text=What this step computes', { timeout: 20000 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: '__pw_nav_pdm.png', fullPage: true });

  console.log('ERRORS_FOUND:' + JSON.stringify(errors));
  await browser.close();
})().catch((e) => { console.error('SCRIPT_FAILED:', e); process.exit(1); });
