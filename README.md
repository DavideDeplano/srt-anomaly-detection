# ⚙️ Setup ambiente 

Il progetto utilizza **Conda** per la gestione dell’ambiente.  
Per creare e configurare automaticamente l’ambiente `srt-anom`.

## 🪟 Per sistemi Windows

```bash
scripts\win\setup_env.bat
```
Questo comando:

- crea o aggiorna l’ambiente Conda `srt-anom` dal file `env.yml`;

- installa tutte le dipendenze necessarie;

- registra il progetto per l’esecuzione diretta.
  
## ▶️ Avvio del progetto

Dopo la configurazione iniziale, per eseguire la pipeline basta:

```bash
scripts\win\run.bat
```

Non è necessario attivare manualmente l’ambiente Conda:
lo script `run.bat` esegue tutto automaticamente nel contesto corretto.

## 🐧🍎 Per sistemi Linux / macOS

Usare gli script equivalenti nella cartella `scripts/unix`:

```bash
scripts/unix/setup_env.sh
scripts/unix/run.sh
```