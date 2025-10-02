// ==UserScript==
// @name         Web Beacon Injector (CNKI/ScienceDirect)
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  Injects web beacon on library e-journal pages for prototype research
// @author       you
// @match        *://*.cnki.net/*
// @match        *://*.cnki.com.cn/*
// @match        *://*.sciencedirect.com/*
// @match        *://*.elsevier.com/*
// @grant        none
// ==/UserScript==
(function() {
  'use strict'
  var BEACON_HOST = (window.BEACON_HOST || 'http://localhost:5000')
  var s = document.createElement('script')
  s.src = BEACON_HOST.replace(/\/$/, '') + '/beacon.js'
  s.async = true
  s.onload = function(){ console.log('[WB] beacon loaded') }
  s.onerror = function(){ console.warn('[WB] beacon failed to load') }
  document.documentElement.appendChild(s)
})();

