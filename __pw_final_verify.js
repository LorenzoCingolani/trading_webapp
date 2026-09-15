const { chromium } = require('playwright');

async function waitIdle(page) {
  try { await page.waitForSelector('text=RUNNING', { state: 'detached', timeout: 120000 }); } catch (e) {}
  await page.waitForTimeout(1000);
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1200 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push('console: ' + msg.text()); });

  await page.goto('http://localhost:8599', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForSelector('text=Trading Analytics', { timeout: 20000 });
  await waitIdle(page);

  // 1) PDM run -> check Generated files section
  await page.getByRole('button', { name: 'PDM', exact: true }).click();
  await waitIdle(page);
  await page.waitForSelector('text=Run PDM', { timeout: 20000 });
  await waitIdle(page);
  await page.getByRole('button', { name: 'Run PDM', exact: true }).click();
  await waitIdle(page);
  await waitIdle(page);
  await page.waitForSelector('text=Generated files', { timeout: 30000 }).catch(() => console.log('Generated files header not found on PDM page'));
  await page.screenshot({ path: '__pw_fv_1_pdm_generated_files.png', fullPage: true });

  // 2) Sharpe Ratio: switch between instruments multiple times, then back, to try to reproduce the mismatch
  await page.getByRole('button', { name: 'Sharpe Ratio', exact: true }).click();
  await waitIdle(page);
  await page.waitForSelector('text=Run Sharpe analysis', { timeout: 20000 });
  await waitIdle(page);
  await page.getByRole('button', { name: 'Run Sharpe analysis', exact: true }).click();
  await waitIdle(page);
  await waitIdle(page);

  const select = page.locator('div[data-baseweb="select"]').first();
  await select.click();
  await waitIdle(page);
  let options = page.locator('li');
  let optCount = await options.count();
  console.log('instrument options:', optCount);
  if (optCount > 1) {
    await options.nth(1).click();
    await waitIdle(page);
    await waitIdle(page);
  }
  await page.screenshot({ path: '__pw_fv_2_sharpe_switched.png', fullPage: true });

  // switch back to first instrument
  await select.click();
  await waitIdle(page);
  options = page.locator('li');
  await options.nth(0).click();
  await waitIdle(page);
  await waitIdle(page);
  await page.screenshot({ path: '__pw_fv_3_sharpe_switched_back.png', fullPage: true });

  console.log('ERRORS_FOUND:' + JSON.stringify(errors));
  await browser.close();
})().catch((e) => { console.error('SCRIPT_FAILED:', e); process.exit(1); });
