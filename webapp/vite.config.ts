import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import tailwindcss from '@tailwindcss/vite';

// Every page except the sign-in page needs a signed-in user (see src/auth.ts): an inline <head>
// script sends signed-out visitors to /login.html?next=... before anything renders or the camera
// is requested. The sign-in page also gets a bare header/footer: the <!--APP-->...<!--/APP-->
// parts of the partials (page links, menus) are removed there, since nothing is reachable yet.
const LOGIN_PAGE = '/login.html';
const AUTH_GUARD =
  "try{if(!localStorage.getItem('hv-session')&&!sessionStorage.getItem('hv-session')){document.documentElement.style.visibility='hidden';location.replace('/login.html?next='+encodeURIComponent(location.pathname+location.search+location.hash))}}catch(e){}";

// Multi-page site: every page shares the header/footer in /partials. The
// <!--HEADER--> / <!--FOOTER--> markers are replaced at build/dev time, and the
// nav link matching the current page is marked active.
function partials(): Plugin {
  return {
    name: 'html-partials',
    transformIndexHtml: {
      order: 'pre',
      handler(html, ctx) {
        const read = (n: string) => readFileSync(resolve(__dirname, 'partials', n), 'utf8');
        const path = ctx.path.split('?')[0];
        const current = path === '/' || path === '/index.html' ? '/index.html' : path;
        const appOnly = (s: string) => (current === LOGIN_PAGE ? s.replace(/<!--APP-->[\s\S]*?<!--\/APP-->/g, '') : s);
        const header = appOnly(read('header.html')).replace(/data-nav="([^"]+)"/g, (_m, href: string) =>
          href === current ? `href="${href}" aria-current="page"` : `href="${href}"`,
        );
        const out = html.replace('<!--HEADER-->', header).replace('<!--FOOTER-->', appOnly(read('footer.html')));
        return {
          html: out,
          tags: [
            ...(current !== LOGIN_PAGE ? [{ tag: 'script', injectTo: 'head-prepend' as const, children: AUTH_GUARD }] : []),
            {
              tag: 'script',
              injectTo: 'head-prepend',
              children:
                "try{var t=localStorage.getItem('theme');if(t!=='dark'&&t!=='light'){t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}document.documentElement.dataset.theme=t}catch(e){}",
            },
          ],
        };
      },
    },
  };
}

export default defineConfig({
  plugins: [tailwindcss(), partials()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        translate: resolve(__dirname, 'translate.html'),
        learn: resolve(__dirname, 'learn.html'),
        math: resolve(__dirname, 'math.html'),
        sentence: resolve(__dirname, 'sentence.html'),
        how: resolve(__dirname, 'how-it-works.html'),
        about: resolve(__dirname, 'about.html'),
        login: resolve(__dirname, 'login.html'),
        history: resolve(__dirname, 'history.html'),
      },
    },
  },
});
