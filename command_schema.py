from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Common configuration
# ---------------------------------------------------------------------------

class StrictModel(BaseModel):
    """
    Base model for all robot commands.

    Extra fields are rejected so the LLM cannot silently introduce
    unsupported parameters.
    """

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Navigation targets
# ---------------------------------------------------------------------------

class NamedLocationTarget(StrictModel):
    type: Literal["named_location"]
    value: str = Field(min_length=1, max_length=100)


Target = NamedLocationTarget


# ---------------------------------------------------------------------------
# Individual robot commands
# ---------------------------------------------------------------------------

class NavigateCommand(StrictModel):
    action: Literal["navigate"]
    target: Target


class MoveCommand(StrictModel):
    action: Literal["move"]

    direction: Literal[
        "forward",
        "backward",
        "left",
        "right",
    ]

    distance: float = Field(gt=0.0, le=10.0)
    unit: Literal["m"]


class RotateCommand(StrictModel):
    action: Literal["rotate"]

    direction: Literal[
        "left",
        "right",
    ]

    angle: float = Field(gt=0.0, le=360.0)
    unit: Literal["deg"]


class StopCommand(StrictModel):
    action: Literal["stop"]


class WaitCommand(StrictModel):
    action: Literal["wait"]

    duration: float = Field(gt=0.0, le=300.0)
    unit: Literal["s"]


# ---------------------------------------------------------------------------
# Sequence command
# ---------------------------------------------------------------------------

class SequenceCommand(StrictModel):
    action: Literal["sequence"]

    commands: list[
        Annotated[
            Union[
                NavigateCommand,
                MoveCommand,
                RotateCommand,
                StopCommand,
                WaitCommand,
            ],
            Field(discriminator="action"),
        ]
    ] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_sequence(self) -> "SequenceCommand":
        """
        Prevent nested sequences in V1.

        Nested task structures can be introduced later when the task
        planner becomes more sophisticated.
        """
        return self


# ---------------------------------------------------------------------------
# Top-level command
# ---------------------------------------------------------------------------

RobotCommand = Annotated[
    Union[
        NavigateCommand,
        MoveCommand,
        RotateCommand,
        StopCommand,
        WaitCommand,
        SequenceCommand,
    ],
    Field(discriminator="action"),
]


def parse_command(data: dict) -> RobotCommand:
    """
    Validate an LLM-generated dictionary against the robot command contract.

    Raises:
        pydantic.ValidationError:
            If the command is invalid.
    """

    # TypeAdapter is the correct Pydantic v2 mechanism for validating
    # a discriminated union that is not itself a BaseModel.
    from pydantic import TypeAdapter

    adapter = TypeAdapter(RobotCommand)

    return adapter.validate_python(data)


def command_to_dict(command: RobotCommand) -> dict:
    """Convert a validated command back to a plain dictionary."""
    return command.model_dump()
