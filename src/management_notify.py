"""TLS email alerts. Addresses/credentials remain in Actions secrets, never reports."""
import hashlib
import json
import os
import smtplib
import ssl
import time
from email.message import EmailMessage
from pathlib import Path


def fingerprint(report):
    # One digest per incident class/day; newly added race IDs do not spam users.
    keys = sorted({f['code']+':'+f.get('message','') for f in report['findings']})
    return hashlib.sha256(json.dumps(keys, ensure_ascii=False).encode()).hexdigest()


def deliver(report, state, now=None, sender=None):
    now = time.time() if now is None else now
    issues = report.get('findings', [])
    if not issues and not state.get('active'):
        return state, 'no_incident'
    key = fingerprint(report) if issues else 'recovered'
    if state.get('fingerprint') == key and now-state.get('sent_at', 0) < 24*3600:
        return state, 'suppressed_duplicate'
    required = ('SMTP_HOST','SMTP_USER','SMTP_PASSWORD','ALERT_FROM','ALERT_TO')
    if not all(os.environ.get(k) for k in required):
        return state, 'not_configured'
    message = EmailMessage()
    message['Subject'] = '[KEIRIN NEXUS] '+('異常 '+str(len(issues))+'件' if issues else '復旧確認')
    message['From'] = os.environ['ALERT_FROM']
    message['To'] = os.environ['ALERT_TO']
    lines = ['検査時刻: '+report['checked_at']]
    for item in issues:
        lines.append(report['owners'].get(item['owner'],item['owner'])+': '+item['message'])
        if item.get('race_ids'):
            lines.append('対象レース: '+', '.join(item['race_ids'][:20]))
        if item.get('run_url'):
            lines.append(item['run_url'])
    lines.append('管理画面: https://rokuda6898-jpg.github.io/keirin-ai-auto/management.html')
    message.set_content('\n'.join(lines))
    if sender:
        sender(message)
    else:
        # Port 465 requires TLS from connection start; all others require STARTTLS.
        port = int(os.environ.get('SMTP_PORT') or '587')
        context = ssl.create_default_context()
        cls = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
        kwargs = {'context':context} if port == 465 else {}
        with cls(os.environ['SMTP_HOST'],port,timeout=30,**kwargs) as client:
            if port != 465:
                client.ehlo()
                client.starttls(context=context)
                client.ehlo()
            client.login(os.environ['SMTP_USER'],os.environ['SMTP_PASSWORD'])
            client.send_message(message)
    return {'fingerprint':key,'sent_at':now,'active':bool(issues)}, 'sent'


if __name__ == '__main__':
    report_path = Path('outputs/management_status.json')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    path = Path('.alert-state/state.json')
    state = json.loads(path.read_text()) if path.exists() else {}
    try:
        state, status = deliver(report, state)
    except Exception as error:
        # Do not log SMTP server text, which can include addresses or credentials.
        report['email_status'] = 'delivery_failed'
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        raise SystemExit('Email delivery failed: '+type(error).__name__)
    report['email_status'] = status
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(state))
    print('Email status: '+status)
