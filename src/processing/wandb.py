class WandbConfiguration:
    team: str
    project: str
    key: str

    def __init__(self, team: str, project: str, key: str):
        self.team = team
        self.project = project
        self.key = key