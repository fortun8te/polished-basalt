# Fork switches

Changes carried only by this fork are switched by the environment variable `BASALT_FORK_<NAME>`
(`basalt::platform::fork_switch`): a change is on unless the variable is `0`. `basalt::platform::fork_number` reads a
number the same way. `fork/windows/Start-ForkBasalt.ps1 -Disable a,b` sets the variables to `0`.

| switch | what it does |
| --- | --- |
