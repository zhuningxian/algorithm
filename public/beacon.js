;(function () {
  'use strict'

  // ===== Config =====
  var CONFIG = {
    endpoint: (window.BEACON_ENDPOINT || 'http://localhost:5000') + '/api/beacon',
    site: window.BEACON_SITE || (location.hostname || 'unknown-site'),
    sendImmediate: true // use navigator.sendBeacon when possible
  }

  // ===== Utilities =====
  function nowMs() { return Date.now() }
  function uuidv4() {
    if (crypto && crypto.randomUUID) return crypto.randomUUID()
    // Fallback
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8)
      return v.toString(16)
    })
  }
  function getOrCreate(key, factory) {
    try {
      var v = localStorage.getItem(key)
      if (v) return v
      var nv = factory()
      localStorage.setItem(key, nv)
      return nv
    } catch (e) {
      // storage disabled
      return factory()
    }
  }
  function pickSearchQueryFromLocation() {
    var params = new URLSearchParams(location.search || '')
    var keys = ['q','query','queryText','qs','wd','word','kw','keys','search','searchTerm','searchword','keyValue']
    for (var i=0;i<keys.length;i++) {
      var val = params.get(keys[i])
      if (val && val.trim().length > 0) return val.trim()
    }
    return null
  }
  function isPdfUrl(href) {
    if (!href) return false
    var u = href.toLowerCase()
    return u.endsWith('.pdf') || u.includes('/pdf') || u.includes('download') || u.includes('full-text') || u.includes('全文')
  }
  function extractArticleId(href) {
    if (!href) return null
    // Try DOI
    var m = href.match(/10\.\d{4,9}\/[-._;()/:A-Z0-9]+/i)
    if (m) return m[0]
    // Else return pathname hash
    try { var url = new URL(href, location.href); return url.pathname } catch(e) { return href }
  }

  // ===== Identity =====
  var userId = (function() {
    if (window.BEACON_USER_ID) return String(window.BEACON_USER_ID)
    return getOrCreate('wb_user_id', function(){ return uuidv4() })
  })()
  var sessionId = (function() {
    return getOrCreate('wb_session_id', function(){ return uuidv4() })
  })()

  // ===== Send function =====
  function send(event) {
    try {
      event.ts = nowMs()
      event.user_id = event.user_id || userId
      event.session_id = event.session_id || sessionId
      event.site = event.site || CONFIG.site
      event.url = event.url || location.href
      event.referrer = event.referrer || document.referrer || null
      event.title = event.title || document.title || null

      var blob = new Blob([JSON.stringify(event)], { type: 'application/json' })
      if (CONFIG.sendImmediate && navigator.sendBeacon) {
        navigator.sendBeacon(CONFIG.endpoint, blob)
        return
      }
      fetch(CONFIG.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(event),
        credentials: 'omit',
        keepalive: true
      }).catch(function(){})
    } catch (e) {
      // swallow
    }
  }

  // ===== page_view =====
  send({ event_type: 'page_view' })

  // ===== time_spent (active time) =====
  var activeStart = nowMs()
  var accumulated = 0
  var visible = !document.hidden
  function onVisibilityChange() {
    var now = nowMs()
    if (document.hidden) {
      if (visible) { accumulated += now - activeStart; visible = false }
    } else {
      if (!visible) { activeStart = now; visible = true }
    }
  }
  document.addEventListener('visibilitychange', onVisibilityChange, { passive: true })
  window.addEventListener('pagehide', function() {
    var now = nowMs()
    var total = accumulated + (visible ? (now - activeStart) : 0)
    if (total > 0) {
      send({ event_type: 'time_spent', dwell_ms: Math.floor(total) })
    }
  })

  // ===== search (form submissions) =====
  function captureFormSubmit(e) {
    try {
      var form = e.target
      if (!form || form.tagName !== 'FORM') return
      var inputs = form.querySelectorAll('input,textarea')
      var q = null
      for (var i=0;i<inputs.length;i++) {
        var el = inputs[i]
        var name = (el.name || '').toLowerCase()
        if (['q','query','querytext','qs','wd','word','kw','keys','search','searchterm','searchword','keyvalue'].indexOf(name) >= 0) {
          q = el.value
          break
        }
        if (el.type === 'search' || el.type === 'text') {
          if (!q && el.value && el.value.trim().length > 0) q = el.value
        }
      }
      if (!q) q = pickSearchQueryFromLocation()
      if (q && q.trim().length > 0) {
        send({ event_type: 'search', search_query: String(q).trim() })
      }
    } catch (err) {}
  }
  document.addEventListener('submit', captureFormSubmit, true)

  // ===== click_article, view_abstract, download (delegated clicks) =====
  function innerText(el) { return (el && (el.innerText || el.textContent || '')).trim() }
  function containsText(el, texts) {
    var t = innerText(el).toLowerCase()
    for (var i=0;i<texts.length;i++) if (t.includes(texts[i])) return true
    return false
  }

  document.addEventListener('click', function(e) {
    try {
      var a = e.target
      while (a && a.tagName && a.tagName !== 'A') a = a.parentElement
      if (!a || a.tagName !== 'A') return
      var href = a.getAttribute('href') || ''
      var text = innerText(a)
      var articleLike = text.length > 20 || (a.rel && a.rel.includes('bookmark')) || (href && /article|doi|record|abs|detail/i.test(href))
      var isAbstract = containsText(a, ['abstract','摘要']) || (href && /#abstract/i.test(href))
      var isDownload = isPdfUrl(href) || containsText(a, ['pdf','download','全文','下载'])

      if (isDownload) {
        send({ event_type: 'download', article_id: extractArticleId(href), article_title: text })
        return
      }
      if (isAbstract) {
        send({ event_type: 'view_abstract', article_id: extractArticleId(href), article_title: text })
        return
      }
      if (articleLike) {
        send({ event_type: 'click_article', article_id: extractArticleId(href), article_title: text })
        return
      }
    } catch (err) {}
  }, true)

  // Fire view_abstract on load if abstract section present
  function checkAbstractOnLoad() {
    try {
      var abs = document.querySelector('#abstract, .abstract, [data-abstract]')
      if (abs) {
        send({ event_type: 'view_abstract', article_id: extractArticleId(location.href), article_title: document.title })
      }
    } catch (e) {}
  }
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(checkAbstractOnLoad, 0)
  } else {
    document.addEventListener('DOMContentLoaded', checkAbstractOnLoad)
  }

  // Also capture search from URL if landing on results page
  var q0 = pickSearchQueryFromLocation()
  if (q0) { send({ event_type: 'search', search_query: q0 }) }
})();

