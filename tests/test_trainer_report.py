import pytest

from tests.test_trainer_models import metrics
from trainer.app import gate, mailing
from trainer.app.config import MailSettings
from trainer.app.report import build_report, fallback_text, unknown_numbers

REPORT = build_report(metrics(error=0.08, base_error=0.10), gate.evaluate(metrics(error=0.08, base_error=0.10), 0.20),
                      0.20, ["laya", "visao"])


def test_report_lists_what_improved_what_was_a_gap_and_what_is_left():
    assert REPORT["melhorou"] == ["assinado", "nao_assinado"]
    assert REPORT["gaps"] == ["nao_assinado"] and REPORT["falta"] == ["nao_assinado"]
    assert REPORT["antes"]["erro"] == 0.10 and REPORT["depois"]["erro"] == 0.08


def test_numbers_match_as_values_or_percentages():
    assert unknown_numbers("O erro caiu de 10% para 8%, uma redução de 20%.", REPORT) == []
    assert unknown_numbers("O erro caiu de 0,10 para 0,08 na versão visao-0002.", REPORT) == []


def test_invented_numbers_are_caught():
    assert unknown_numbers("O erro caiu 47% e chegou a 3%.", REPORT) == ["47", "3"]


def test_fallback_text_only_uses_numbers_from_the_report():
    assert unknown_numbers(fallback_text(REPORT), REPORT) == []


class Writer:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), 0

    def write(self, report):
        self.calls += 1
        item = self.texts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_a_faithful_narrative_is_sent_as_written():
    text = "O erro caiu de 10% para 8%."
    assert mailing.compose(REPORT, Writer(text)) == text


def test_a_narrative_with_invented_numbers_is_retried_then_replaced_by_the_fallback():
    writer = Writer("O erro caiu 47%.", "Agora está em 3%.")
    assert mailing.compose(REPORT, writer) == fallback_text(REPORT) and writer.calls == mailing.MAX_WRITE_ATTEMPTS


def test_hermes_down_falls_back_to_the_plain_text():
    writer = Writer(mailing.WriterError("fora do ar"), mailing.WriterError("fora do ar"))
    assert mailing.compose(REPORT, writer) == fallback_text(REPORT)


class FakeSmtp:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.log.append("starttls")

    def login(self, user, password):
        self.log.append(("login", user))

    def send_message(self, message):
        self.log.append(("mail", message["To"], message["Subject"]))


def test_mailer_sends_through_starttls_with_login():
    log = []
    settings = MailSettings("smtp.local", 587, "u", "p", "laya@x", ("a@x", "b@x"))
    mailing.Mailer(settings, lambda host, port, timeout: FakeSmtp(log)).send("assunto", "corpo")
    assert log == ["starttls", ("login", "u"), ("mail", "a@x, b@x", "assunto")]


def test_mailer_failure_is_a_typed_error():
    def broken(host, port, timeout):
        raise OSError("sem rota")

    with pytest.raises(mailing.MailError):
        mailing.Mailer(MailSettings("smtp.local", 587, "", "", "laya@x", ("a@x",)), broken).send("a", "b")


def test_mail_is_disabled_without_host_sender_and_recipients():
    assert not MailSettings().enabled and MailSettings("h", 465, "", "", "f@x", ("t@x",)).enabled
